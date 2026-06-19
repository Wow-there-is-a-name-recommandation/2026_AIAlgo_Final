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
        self.prototypes = nn.Parameter(torch.randn(num_layers, num_classes, embed_dim) * 0.02)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bld,lcd->blc", z, p)
        logits = sim.mean(dim=1)
        return logits, sim, z, p


class LayerWiseConceptBottleneck(nn.Module):
    def __init__(self, num_layers=32, num_concepts=2, hidden_dim=64):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(num_layers * num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, layerwise_sim):
        x = layerwise_sim.flatten(start_dim=1)
        concept_logits = self.bottleneck(x)
        concept_score = torch.sigmoid(concept_logits)
        return concept_logits, concept_score


def load_layerwise_cbm(cbm_path, device):
    ckpt = torch.load(cbm_path, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

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
    global CBM_TRACE, LAYER_STATES
    CBM_TRACE = []
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


def layerwise_cbm_hook(module, args, layer_idx, cbm_proto, cbm, inv_label_map, num_layers):
    global CBM_TRACE, LAYER_STATES

    hidden_states = args[0]

    # 각 layer의 마지막 token hidden 저장
    LAYER_STATES[layer_idx] = hidden_states[:, -1, :].detach().float()

    # 마지막 layer에서 CBM 분석
    if layer_idx == num_layers - 1:
        if all(i in LAYER_STATES for i in range(num_layers)):
            layer_stack = torch.stack([LAYER_STATES[i] for i in range(num_layers)], dim=1)

            with torch.no_grad():
                _, layerwise_sim, _, _ = cbm_proto(layer_stack)
                concept_logits, concept_score = cbm(layerwise_sim)
                concept_prob = torch.softmax(concept_logits, dim=-1)

            label_to_id = {v: k for k, v in inv_label_map.items()}

            for b in range(hidden_states.shape[0]):
                score_dict = {
                    inv_label_map[i]: float(concept_score[b, i].detach().cpu())
                    for i in range(concept_score.shape[-1])
                }
                prob_dict = {
                    inv_label_map[i]: float(concept_prob[b, i].detach().cpu())
                    for i in range(concept_prob.shape[-1])
                }
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

                if "existence" in label_to_id:
                    ex_id = label_to_id["existence"]
                    existence_layer_sim = [
                        float(v) for v in layerwise_sim[b, :, ex_id].detach().cpu().tolist()
                    ]

                if "safe" in label_to_id:
                    safe_id = label_to_id["safe"]
                    safe_layer_sim = [
                        float(v) for v in layerwise_sim[b, :, safe_id].detach().cpu().tolist()
                    ]

                pred_id = int(torch.argmax(concept_prob[b]).item())

                CBM_TRACE.append({
                    "batch_index": b,
                    "layer": "all",
                    "router_action": None,
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

                    "concept_score": score_dict,
                    "concept_prob": prob_dict,
                    "avg_prototype_similarity": avg_sim_dict,
                    "max_prototype_similarity": max_sim_dict,
                })

        LAYER_STATES = {}

    return (hidden_states,)


def register_hooks(model, cbm_proto, cbm, inv_label_map, num_layers):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    if len(layers) != num_layers:
        print(f"Warning: model has {len(layers)} layers but checkpoint num_layers={num_layers}")

    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(
            layerwise_cbm_hook,
            layer_idx=i,
            cbm_proto=cbm_proto,
            cbm=cbm,
            inv_label_map=inv_label_map,
            num_layers=num_layers,
        )
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)

    print(f"Registered LLaVA + layer-wise existence CBM hooks to {len(hooks)} layers.")
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

    parser.add_argument("--cbm-path", type=str, required=True)

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--anno-folder", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-size", type=int, default=500)
    parser.add_argument("--output-file", type=str, default="output/gen_cap_llava_layerwise_exis_cbm_analysis.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--risk-threshold", type=float, default=0.5)

    args = parser.parse_args()
    eval_model(args)