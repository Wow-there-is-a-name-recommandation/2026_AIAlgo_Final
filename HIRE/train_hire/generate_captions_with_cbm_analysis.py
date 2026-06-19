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


class PrototypeEncoder(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.prototypes = nn.Parameter(torch.randn(num_classes, embed_dim) * 0.02)

    def forward(self, x):
        z = F.normalize(self.encoder(x), dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        proto_sim = z @ p.t()
        return proto_sim, z, p


class ConceptBottleneck(nn.Module):
    def __init__(self, num_concepts=4, hidden_dim=64):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, proto_sim):
        concept_logits = self.bottleneck(proto_sim)
        concept_score = torch.sigmoid(concept_logits)
        return concept_logits, concept_score


def load_cbm(cbm_path, device):
    ckpt = torch.load(cbm_path, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    prototype_args = ckpt.get("prototype_args", {})
    cbm_args = ckpt.get("args", {})

    proto = PrototypeEncoder(
        input_dim=4096,
        embed_dim=prototype_args.get("embed_dim", 256),
        hidden_dim=prototype_args.get("hidden_dim", 1024),
        num_classes=len(label_map),
    )

    cbm = ConceptBottleneck(
        num_concepts=len(label_map),
        hidden_dim=cbm_args.get("cbm_hidden_dim", 64),
    )

    proto.load_state_dict(ckpt["prototype"])
    cbm.load_state_dict(ckpt["cbm"])

    proto.to(device).eval()
    cbm.to(device).eval()

    return proto, cbm, label_map, inv_label_map


def clear_cbm_trace():
    global CBM_TRACE, CURRENT_STEP_ACTION
    CBM_TRACE = []
    CURRENT_STEP_ACTION = None


def editor_router_cbm_hook(
    module,
    args,
    layer_idx,
    editor,
    agent,
    cbm_proto,
    cbm,
    inv_label_map,
    analysis_layer,
):
    global CURRENT_STEP_ACTION, CBM_TRACE

    hidden_states = args[0]

    # layer 0: router action 결정 + CBM analysis state 저장
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

    # CBM analysis는 지정 layer에서만 수행
    # 기본값 layer 0: router와 동일한 state_input 사용
    if layer_idx == analysis_layer:
        state_input = hidden_states[:, -1, :].detach().float()

        with torch.no_grad():
            proto_sim, _, _ = cbm_proto(state_input)
            concept_logits, concept_score = cbm(proto_sim)
            concept_prob = torch.softmax(concept_logits, dim=-1)

        for b in range(hidden_states.shape[0]):
            score_dict = {
                inv_label_map[i]: float(concept_score[b, i].detach().cpu())
                for i in range(concept_score.shape[-1])
            }
            prob_dict = {
                inv_label_map[i]: float(concept_prob[b, i].detach().cpu())
                for i in range(concept_prob.shape[-1])
            }
            sim_dict = {
                inv_label_map[i]: float(proto_sim[b, i].detach().cpu())
                for i in range(proto_sim.shape[-1])
            }

            pred_id = int(torch.argmax(concept_prob[b]).item())
            router_action = None
            if CURRENT_STEP_ACTION is not None:
                router_action = int(CURRENT_STEP_ACTION[b].detach().cpu().item())

            CBM_TRACE.append({
                "batch_index": b,
                "layer": layer_idx,
                "router_action": router_action,
                "pred_concept": inv_label_map[pred_id],
                "concept_score": score_dict,
                "concept_prob": prob_dict,
                "prototype_similarity": sim_dict,
            })

    # 기존 HIRE editor 적용
    if editor is not None and getattr(editor, "training", True) is False:
        if layer_idx * 2 in editor.train_layer and CURRENT_STEP_ACTION is not None:
            edit_mask = CURRENT_STEP_ACTION.bool()

            editor.cur_layer_id = f"{layer_idx}.attn"
            edited_states = editor.edit(hidden_states)

            hidden_states[edit_mask] = edited_states[edit_mask]

    return (hidden_states,)


def register_hooks(model, editor, agent, cbm_proto, cbm, inv_label_map, analysis_layer):
    hooks = []
    layers = model.model.layers if hasattr(model, "model") else model.layers

    for i, layer in enumerate(layers):
        target_module = layer.self_attn.o_proj
        hook_fn = partial(
            editor_router_cbm_hook,
            layer_idx=i,
            editor=editor,
            agent=agent,
            cbm_proto=cbm_proto,
            cbm=cbm,
            inv_label_map=inv_label_map,
            analysis_layer=analysis_layer,
        )
        handle = target_module.register_forward_pre_hook(hook_fn)
        hooks.append(handle)

    print(f"Registered HIRE + CBM analysis hooks to {len(hooks)} layers.")
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
        print("Router path is None. All tokens will be edited by default, same as original fallback.")

    cbm_proto, cbm, label_map, inv_label_map = load_cbm(args.cbm_path, device)
    print("Loaded CBM:", args.cbm_path)
    print("CBM label_map:", label_map)

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
        analysis_layer=args.analysis_layer,
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

                result = {
                    "image_id": img_id,
                    "image_name": image_name,
                    "caption": caption,
                    "cbm_analysis": grouped_trace[idx],
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
    parser.add_argument("--data-size", type=int, default=2000)
    parser.add_argument("--output-file", type=str, default="output/gen_cap_cbm_analysis.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")

    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--edit-strength", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)

    # CBM을 어느 layer hidden으로 분석할지.
    # 0이면 기존 router state와 동일 위치.
    parser.add_argument("--analysis-layer", type=int, default=0)

    args = parser.parse_args()
    eval_model(args)