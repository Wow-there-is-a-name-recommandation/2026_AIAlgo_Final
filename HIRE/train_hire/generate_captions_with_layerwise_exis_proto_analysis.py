import argparse
import json
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from filelock import FileLock

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path

from torch.utils.data import DataLoader, DistributedSampler
from coco_data import CocoDataset, custom_collate_fn
from truthx import TruthX
from router import DPOAgent


CURRENT_STEP_ACTION = None
CBM_TRACE = []
LAYER_STATES = {}


class LayerWisePrototypeEncoder(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=2, num_layers=32):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.prototypes = nn.Parameter(
            torch.randn(num_layers, num_classes, embed_dim) * 0.02
        )

    def forward(self, x):
        # x: [B,L,H]
        z = F.normalize(self.encoder(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bld,lcd->blc", z, p)  # [B,L,C]
        logits = sim.mean(dim=1)
        return logits, sim, z, p


def load_layerwise_prototype(prototype_path, device):
    ckpt = torch.load(prototype_path, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    if "existence" not in label_map or "safe" not in label_map:
        raise ValueError(f"This script expects existence/safe label_map, got {label_map}")

    proto_args = ckpt.get("args", {})
    num_layers = ckpt.get("num_layers", proto_args.get("num_layers", 32))

    proto = LayerWisePrototypeEncoder(
        input_dim=4096,
        embed_dim=proto_args.get("embed_dim", 256),
        hidden_dim=proto_args.get("hidden_dim", 1024),
        num_classes=len(label_map),
        num_layers=num_layers,
    )

    proto.load_state_dict(ckpt["model"])
    proto.to(device).eval()

    return proto, label_map, inv_label_map, num_layers


def clear_cbm_trace():
    global CBM_TRACE, CURRENT_STEP_ACTION, LAYER_STATES
    CBM_TRACE = []
    CURRENT_STEP_ACTION = None
    LAYER_STATES = {}


def summarize_existence_trace(trace, threshold=0.5):
    scores, probs, sims = [], [], []

    for item in trace:
        scores.append(item.get("existence_score", 0.0))
        probs.append(item.get("existence_prob", 0.0))
        sims.append(item.get("avg_existence_proto_sim", 0.0))

    if not scores:
        return {
            "num_steps": 0,
            "avg_existence_score": 0.0,
            "max_existence_score": 0.0,
            "top5_existence_score": 0.0,
            "existence_risk_ratio": 0.0,
            "avg_existence_prob": 0.0,
            "max_existence_prob": 0.0,
            "avg_existence_proto_sim": 0.0,
            "max_existence_proto_sim": 0.0,
        }

    top5 = sorted(scores, reverse=True)[:5]

    return {
        "num_steps": len(scores),
        "avg_existence_score": float(sum(scores) / len(scores)),
        "max_existence_score": float(max(scores)),
        "top5_existence_score": float(sum(top5) / len(top5)),
        "existence_risk_ratio": float(sum(s > threshold for s in scores) / len(scores)),
        "avg_existence_prob": float(sum(probs) / len(probs)),
        "max_existence_prob": float(max(probs)),
        "avg_existence_proto_sim": float(sum(sims) / len(sims)),
        "max_existence_proto_sim": float(max(sims)),
    }


def editor_router_layerwise_proto_hook(
    module,
    args,
    layer_idx,
    editor,
    agent,
    proto,
    inv_label_map,
    num_layers,
):
    global CURRENT_STEP_ACTION, CBM_TRACE, LAYER_STATES

    hidden_states = args[0]

    if layer_idx == 0:
        ori_dtype = hidden_states.dtype
        state_input = hidden_states[:, -1, :].detach().float()

        if agent is None:
            actions = torch.ones((hidden_states.shape[0],), dtype=ori_dtype, device=hidden_states.device)
        else:
            with torch.no_grad():
                logits = agent.actor(state_input)
                actions = torch.argmax(logits, dim=-1).to(ori_dtype)

        CURRENT_STEP_ACTION = actions
        LAYER_STATES = {}

    LAYER_STATES[layer_idx] = hidden_states[:, -1, :].detach().float()

    if layer_idx == num_layers - 1:
        if all(i in LAYER_STATES for i in range(num_layers)):
            layer_stack = torch.stack([LAYER_STATES[i] for i in range(num_layers)], dim=1)

            with torch.no_grad():
                _, layerwise_sim, _, _ = proto(layer_stack)  # [B,L,C]

                # 핵심 변경: MLP-CBM 없이 prototype similarity를 concept activation으로 직접 변환
                layer_concept_score = torch.softmax(layerwise_sim, dim=-1)  # [B,L,C]
                concept_score = layer_concept_score.mean(dim=1)             # [B,C]
                concept_prob = concept_score

            concept_to_id = {v: k for k, v in inv_label_map.items()}

            for b in range(hidden_states.shape[0]):
                score_dict = {
                    inv_label_map[i]: float(concept_score[b, i].detach().cpu())
                    for i in range(concept_score.shape[-1])
                }
                prob_dict = dict(score_dict)

                avg_sim_dict = {
                    inv_label_map[i]: float(layerwise_sim[b, :, i].mean().detach().cpu())
                    for i in range(layerwise_sim.shape[-1])
                }
                max_sim_dict = {
                    inv_label_map[i]: float(layerwise_sim[b, :, i].max().detach().cpu())
                    for i in range(layerwise_sim.shape[-1])
                }

                existence_layer_sim = []
                safe_layer_sim = []
                existence_layer_score = []
                safe_layer_score = []

                if "existence" in concept_to_id:
                    ex_id = concept_to_id["existence"]
                    existence_layer_sim = [float(v) for v in layerwise_sim[b, :, ex_id].detach().cpu().tolist()]
                    existence_layer_score = [float(v) for v in layer_concept_score[b, :, ex_id].detach().cpu().tolist()]

                if "safe" in concept_to_id:
                    safe_id = concept_to_id["safe"]
                    safe_layer_sim = [float(v) for v in layerwise_sim[b, :, safe_id].detach().cpu().tolist()]
                    safe_layer_score = [float(v) for v in layer_concept_score[b, :, safe_id].detach().cpu().tolist()]

                pred_id = int(torch.argmax(concept_score[b]).item())

                router_action = None
                if CURRENT_STEP_ACTION is not None:
                    router_action = int(CURRENT_STEP_ACTION[b].detach().cpu().item())

                CBM_TRACE.append({
                    "batch_index": b,
                    "layer": "all",
                    "router_action": router_action,
                    "pred_concept": inv_label_map[pred_id],

                    "existence_score": score_dict.get("existence", 0.0),
                    "safe_score": score_dict.get("safe", 0.0),
                    "existence_prob": prob_dict.get("existence", 0.0),
                    "safe_prob": prob_dict.get("safe", 0.0),

                    "avg_existence_proto_sim": avg_sim_dict.get("existence", 0.0),
                    "avg_safe_proto_sim": avg_sim_dict.get("safe", 0.0),
                    "max_existence_proto_sim": max_sim_dict.get("existence", 0.0),
                    "max_safe_proto_sim": max_sim_dict.get("safe", 0.0),

                    "existence_layer_proto_sim": existence_layer_sim,
                    "safe_layer_proto_sim": safe_layer_sim,
                    "existence_layer_score": existence_layer_score,
                    "safe_layer_score": safe_layer_score,

                    "concept_score": score_dict,
                    "concept_prob": prob_dict,
                    "avg_prototype_similarity": avg_sim_dict,
                    "max_prototype_similarity": max_sim_dict,
                    "score_source": "prototype_softmax_no_mlp",
                })

        LAYER_STATES = {}

    if editor is not None and getattr(editor, "training", True) is False:
        if layer_idx * 2 in editor.train_layer and CURRENT_STEP_ACTION is not None:
            edit_mask = CURRENT_STEP_ACTION.bool()

            editor.cur_layer_id = f"{layer_idx}.attn"
            edited_states = editor.edit(hidden_states)

            hidden_states[edit_mask] = edited_states[edit_mask]

    return (hidden_states,)


def register_hooks(model, editor, agent, proto, inv_label_map, num_layers):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    if len(layers) != num_layers:
        print(f"Warning: model has {len(layers)} layers but checkpoint num_layers={num_layers}")

    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(
            editor_router_layerwise_proto_hook,
            layer_idx=i,
            editor=editor,
            agent=agent,
            proto=proto,
            inv_label_map=inv_label_map,
            num_layers=num_layers,
        )
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)

    print(f"Registered HIRE + layer-wise prototype activation hooks to {len(hooks)} layers.")
    return hooks


def group_trace_by_batch(trace, batch_size):
    grouped = [[] for _ in range(batch_size)]
    step_counter = [0 for _ in range(batch_size)]

    for item in trace:
        b = item["batch_index"]
        if b >= batch_size:
            continue
        item = dict(item)
        item["step"] = step_counter[b]
        grouped[b].append(item)
        step_counter[b] += 1

    return grouped


def eval_model(args):
    disable_torch_init()
    device = "cuda:0"

    train_layer = list(range(0, 63, 2))

    editor = TruthX(args.editor_path, 4096, [2048, 1024], train_layer=train_layer, device=device)
    editor.eval()
    editor.edit_strength = args.edit_strength

    if args.router_path:
        agent = DPOAgent(hidden_dim=4096, latent_dim=[2048, 1024], action_dim=2, device=device)
        checkpoint = torch.load(args.router_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["state_dict"])
        agent.eval()
        print(f"Loaded Router from {args.router_path}")
    else:
        agent = None
        print("Router path is None. All tokens will be edited by default.")

    proto, label_map, inv_label_map, num_layers = load_layerwise_prototype(args.prototype_path, device)
    print("Loaded layer-wise prototype:", args.prototype_path)
    print("Prototype label_map:", label_map)
    print("num_layers:", num_layers)

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path,
        None,
        model_name,
        device=device,
        device_map=device,
    )

    register_hooks(
        model=model,
        editor=editor,
        agent=agent,
        proto=proto,
        inv_label_map=inv_label_map,
        num_layers=num_layers,
    )

    dataset = CocoDataset(
        args.image_folder,
        args.anno_folder,
        subset_size=args.data_size,
        seed=args.seed,
    )

    sampler = DistributedSampler(
        dataset,
        num_replicas=args.num_chunks,
        rank=args.chunk_idx,
        seed=args.seed,
        shuffle=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=custom_collate_fn,
        drop_last=False,
    )

    output_file = os.path.expanduser(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "a", encoding="utf-8") as out_file:
        for image, image_names, image_path in tqdm(dataloader):
            qs = DEFAULT_IMAGE_TOKEN + "\nPlease describe this image in detail."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(
                prompt,
                tokenizer,
                IMAGE_TOKEN_INDEX,
                return_tensors="pt",
            ).unsqueeze(0).to(device)

            input_ids = input_ids.repeat(len(image), 1)
            image_tensor = process_images(image, image_processor, model.config)

            clear_cbm_trace()

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().to(device),
                    do_sample=False,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            grouped_trace = group_trace_by_batch(CBM_TRACE, len(image_names))

            results = []

            for idx, (image_name, text) in enumerate(zip(image_names, outputs)):
                img_id = int(image_name.split("_")[-1].split(".")[0])
                caption = text.split("ASSISTANT:")[-1].strip()

                analysis = grouped_trace[idx]
                exis_summary = summarize_existence_trace(
                    analysis,
                    threshold=args.risk_threshold,
                )

                result = {
                    "image_id": img_id,
                    "image_name": image_name,
                    "caption": caption,
                    "existence_summary": exis_summary,
                    "cbm_analysis": analysis,
                    "analysis_type": "layerwise_prototype_activation_no_mlp",
                }

                results.append(json.dumps(result, ensure_ascii=False))

            with FileLock("000.lock"):
                out_file.write("\n".join(results) + "\n")
                out_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--editor-path", type=str, required=True)
    parser.add_argument("--router-path", type=str, default=None)
    parser.add_argument("--prototype-path", type=str, required=True)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--anno-folder", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=500)
    parser.add_argument("--output-file", type=str, default="output/gen_cap_layerwise_proto_analysis.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--risk-threshold", type=float, default=0.5)

    args = parser.parse_args()
    eval_model(args)