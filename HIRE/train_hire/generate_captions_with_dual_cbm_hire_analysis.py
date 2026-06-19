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
PRE_LAYER_STATES = {}
POST_LAYER_STATES = {}


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
        self.prototypes = nn.Parameter(torch.randn(num_layers, num_classes, embed_dim) * 0.02)

    def forward(self, x):
        # x: [B,L,H]
        z = F.normalize(self.encoder(x), dim=-1)          # [B,L,D]
        p = F.normalize(self.prototypes, dim=-1)          # [L,C,D]
        sim = torch.einsum("bld,lcd->blc", z, p)          # [B,L,C]
        logits = sim.mean(dim=1)                          # [B,C]
        return logits, sim, z, p


class LayerWiseConceptBottleneck(nn.Module):
    def __init__(self, num_layers=32, num_concepts=2, hidden_dim=64):
        super().__init__()
        self.num_layers = num_layers
        self.num_concepts = num_concepts
        self.bottleneck = nn.Sequential(
            nn.Linear(num_layers * num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, layerwise_sim):
        # layerwise_sim: [B,L,C]
        x = layerwise_sim.flatten(start_dim=1)  # [B,L*C]
        concept_logits = self.bottleneck(x)
        concept_score = torch.sigmoid(concept_logits)
        return concept_logits, concept_score


def load_layerwise_cbm(cbm_path, device):
    ckpt = torch.load(cbm_path, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    if "existence" not in label_map or "safe" not in label_map:
        raise ValueError(f"This script expects existence/safe label_map, got {label_map}")

    prototype_args = ckpt.get("prototype_args", {})
    cbm_args = ckpt.get("args", {})

    num_layers = ckpt.get(
        "num_layers",
        prototype_args.get("num_layers", cbm_args.get("num_layers", 32)),
    )

    proto = LayerWisePrototypeEncoder(
        input_dim=4096,
        embed_dim=prototype_args.get("embed_dim", 256),
        hidden_dim=prototype_args.get("hidden_dim", 1024),
        num_classes=len(label_map),
        num_layers=num_layers,
    )

    cbm = LayerWiseConceptBottleneck(
        num_layers=num_layers,
        num_concepts=len(label_map),
        hidden_dim=cbm_args.get("cbm_hidden_dim", 64),
    )

    proto.load_state_dict(ckpt["prototype"])
    cbm.load_state_dict(ckpt["cbm"])

    proto.to(device).eval()
    cbm.to(device).eval()

    return proto, cbm, label_map, inv_label_map, num_layers


def clear_cbm_trace():
    global CBM_TRACE, CURRENT_STEP_ACTION, PRE_LAYER_STATES, POST_LAYER_STATES
    CBM_TRACE = []
    CURRENT_STEP_ACTION = None
    PRE_LAYER_STATES = {}
    POST_LAYER_STATES = {}

def run_layerwise_cbm(layer_states, cbm_proto, cbm, num_layers):
    if not all(i in layer_states for i in range(num_layers)):
        return None

    layer_stack = torch.stack([layer_states[i] for i in range(num_layers)], dim=1)

    with torch.no_grad():
        _, layerwise_sim, _, _ = cbm_proto(layer_stack)
        concept_logits, concept_score = cbm(layerwise_sim)
        concept_prob = torch.softmax(concept_logits, dim=-1)

    return layerwise_sim, concept_score, concept_prob


def build_dual_cbm_item(
    b,
    pre_sim,
    pre_score,
    pre_prob,
    post_sim,
    post_score,
    post_prob,
    inv_label_map,
    router_action,
):
    label_to_id = {v: k for k, v in inv_label_map.items()}
    ex_id = label_to_id["existence"]
    safe_id = label_to_id["safe"]

    pre_pred = int(torch.argmax(pre_prob[b]).item())
    post_pred = int(torch.argmax(post_prob[b]).item())

    return {
        "batch_index": b,
        "layer": "all",
        "router_action": router_action,

        "pre_pred_concept": inv_label_map[pre_pred],
        "post_pred_concept": inv_label_map[post_pred],

        "pre_existence_score": float(pre_score[b, ex_id].detach().cpu()),
        "post_existence_score": float(post_score[b, ex_id].detach().cpu()),
        "delta_existence_score": float((post_score[b, ex_id] - pre_score[b, ex_id]).detach().cpu()),

        "pre_safe_score": float(pre_score[b, safe_id].detach().cpu()),
        "post_safe_score": float(post_score[b, safe_id].detach().cpu()),
        "delta_safe_score": float((post_score[b, safe_id] - pre_score[b, safe_id]).detach().cpu()),

        "pre_existence_prob": float(pre_prob[b, ex_id].detach().cpu()),
        "post_existence_prob": float(post_prob[b, ex_id].detach().cpu()),
        "delta_existence_prob": float((post_prob[b, ex_id] - pre_prob[b, ex_id]).detach().cpu()),

        "pre_safe_prob": float(pre_prob[b, safe_id].detach().cpu()),
        "post_safe_prob": float(post_prob[b, safe_id].detach().cpu()),
        "delta_safe_prob": float((post_prob[b, safe_id] - pre_prob[b, safe_id]).detach().cpu()),

        "pre_avg_existence_proto_sim": float(pre_sim[b, :, ex_id].mean().detach().cpu()),
        "post_avg_existence_proto_sim": float(post_sim[b, :, ex_id].mean().detach().cpu()),
        "delta_avg_existence_proto_sim": float((post_sim[b, :, ex_id].mean() - pre_sim[b, :, ex_id].mean()).detach().cpu()),

        "pre_avg_safe_proto_sim": float(pre_sim[b, :, safe_id].mean().detach().cpu()),
        "post_avg_safe_proto_sim": float(post_sim[b, :, safe_id].mean().detach().cpu()),
        "delta_avg_safe_proto_sim": float((post_sim[b, :, safe_id].mean() - pre_sim[b, :, safe_id].mean()).detach().cpu()),

        "pre_existence_layer_proto_sim": [float(v) for v in pre_sim[b, :, ex_id].detach().cpu().tolist()],
        "post_existence_layer_proto_sim": [float(v) for v in post_sim[b, :, ex_id].detach().cpu().tolist()],
        "pre_safe_layer_proto_sim": [float(v) for v in pre_sim[b, :, safe_id].detach().cpu().tolist()],
        "post_safe_layer_proto_sim": [float(v) for v in post_sim[b, :, safe_id].detach().cpu().tolist()],
    }

def summarize_existence_trace(trace, threshold=0.5):
    pre_scores, post_scores = [], []
    pre_probs, post_probs = [], []
    delta_scores, delta_probs = [], []
    pre_safe_scores, post_safe_scores = [], []
    delta_safe_scores = []

    for item in trace:
        pre_scores.append(item.get("pre_existence_score", 0.0))
        post_scores.append(item.get("post_existence_score", 0.0))
        delta_scores.append(item.get("delta_existence_score", 0.0))

        pre_probs.append(item.get("pre_existence_prob", 0.0))
        post_probs.append(item.get("post_existence_prob", 0.0))
        delta_probs.append(item.get("delta_existence_prob", 0.0))

        pre_safe_scores.append(item.get("pre_safe_score", 0.0))
        post_safe_scores.append(item.get("post_safe_score", 0.0))
        delta_safe_scores.append(item.get("delta_safe_score", 0.0))

    if not pre_scores:
        return {
            "num_steps": 0,
            "pre_avg_existence_score": 0.0,
            "post_avg_existence_score": 0.0,
            "delta_avg_existence_score": 0.0,
            "pre_avg_safe_score": 0.0,
            "post_avg_safe_score": 0.0,
            "delta_avg_safe_score": 0.0,
        }

    return {
        "num_steps": len(pre_scores),

        "pre_avg_existence_score": float(sum(pre_scores) / len(pre_scores)),
        "post_avg_existence_score": float(sum(post_scores) / len(post_scores)),
        "delta_avg_existence_score": float(sum(delta_scores) / len(delta_scores)),

        "pre_max_existence_score": float(max(pre_scores)),
        "post_max_existence_score": float(max(post_scores)),

        "pre_existence_risk_ratio": float(sum(s > threshold for s in pre_scores) / len(pre_scores)),
        "post_existence_risk_ratio": float(sum(s > threshold for s in post_scores) / len(post_scores)),

        "pre_avg_existence_prob": float(sum(pre_probs) / len(pre_probs)),
        "post_avg_existence_prob": float(sum(post_probs) / len(post_probs)),
        "delta_avg_existence_prob": float(sum(delta_probs) / len(delta_probs)),

        "pre_avg_safe_score": float(sum(pre_safe_scores) / len(pre_safe_scores)),
        "post_avg_safe_score": float(sum(post_safe_scores) / len(post_safe_scores)),
        "delta_avg_safe_score": float(sum(delta_safe_scores) / len(delta_safe_scores)),
    }


def editor_router_layerwise_cbm_hook(
    module,
    args,
    layer_idx,
    editor,
    agent,
    cbm_proto,
    cbm,
    inv_label_map,
    num_layers,
):
    global CURRENT_STEP_ACTION, CBM_TRACE, PRE_LAYER_STATES, POST_LAYER_STATES

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
        PRE_LAYER_STATES = {}
        POST_LAYER_STATES = {}

    # 1) HIRE edit 전 표현 저장
    PRE_LAYER_STATES[layer_idx] = hidden_states[:, -1, :].detach().float()

    # 2) HIRE editor 적용
    if editor is not None and getattr(editor, "training", True) is False:
        if layer_idx * 2 in editor.train_layer and CURRENT_STEP_ACTION is not None:
            edit_mask = CURRENT_STEP_ACTION.bool()

            editor.cur_layer_id = f"{layer_idx}.attn"
            edited_states = editor.edit(hidden_states)

            hidden_states[edit_mask] = edited_states[edit_mask]

    # 3) HIRE edit 후 표현 저장
    POST_LAYER_STATES[layer_idx] = hidden_states[:, -1, :].detach().float()

    # 4) 마지막 layer에서 pre/post 표현을 각각 CBM 분석
    if layer_idx == num_layers - 1:
        pre_result = run_layerwise_cbm(PRE_LAYER_STATES, cbm_proto, cbm, num_layers)
        post_result = run_layerwise_cbm(POST_LAYER_STATES, cbm_proto, cbm, num_layers)

        if pre_result is not None and post_result is not None:
            pre_sim, pre_score, pre_prob = pre_result
            post_sim, post_score, post_prob = post_result

            for b in range(hidden_states.shape[0]):
                router_action = None
                if CURRENT_STEP_ACTION is not None:
                    router_action = int(CURRENT_STEP_ACTION[b].detach().cpu().item())

                CBM_TRACE.append(
                    build_dual_cbm_item(
                        b=b,
                        pre_sim=pre_sim,
                        pre_score=pre_score,
                        pre_prob=pre_prob,
                        post_sim=post_sim,
                        post_score=post_score,
                        post_prob=post_prob,
                        inv_label_map=inv_label_map,
                        router_action=router_action,
                    )
                )

        PRE_LAYER_STATES = {}
        POST_LAYER_STATES = {}

    return (hidden_states,)


def register_hooks(model, editor, agent, cbm_proto, cbm, inv_label_map, num_layers):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    if len(layers) != num_layers:
        print(f"Warning: model has {len(layers)} layers but checkpoint num_layers={num_layers}")

    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(
            editor_router_layerwise_cbm_hook,
            layer_idx=i,
            editor=editor,
            agent=agent,
            cbm_proto=cbm_proto,
            cbm=cbm,
            inv_label_map=inv_label_map,
            num_layers=num_layers,
        )
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)

    print(f"Registered HIRE + layer-wise existence CBM hooks to {len(hooks)} layers.")
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

    cbm_proto, cbm, label_map, inv_label_map, num_layers = load_layerwise_cbm(args.cbm_path, device)
    print("Loaded layer-wise existence CBM:", args.cbm_path)
    print("CBM label_map:", label_map)
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
        cbm_proto=cbm_proto,
        cbm=cbm,
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

                cbm_analysis = grouped_trace[idx]
                exis_summary = summarize_existence_trace(
                    cbm_analysis,
                    threshold=args.risk_threshold,
                )

                result = {
                    "image_id": img_id,
                    "image_name": image_name,
                    "caption": caption,
                    "existence_summary": exis_summary,
                    "cbm_analysis": cbm_analysis,
                }

                results.append(json.dumps(result, ensure_ascii=False))

            with FileLock("000.lock"):
                out_file.write("\n".join(results) + "\n")
                out_file.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--editor-path", type=str, required=True)
    parser.add_argument("--router-path", type=str, default=None)
    parser.add_argument("--cbm-path", type=str, required=True)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--anno-folder", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=500)
    parser.add_argument("--output-file", type=str, default="output/gen_cap_layerwise_exis_cbm_analysis.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    parser.add_argument("--risk-threshold", type=float, default=0.5)

    args = parser.parse_args()
    eval_model(args)