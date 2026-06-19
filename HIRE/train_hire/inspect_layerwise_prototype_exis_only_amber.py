import argparse
import glob
import json
import os
from collections import defaultdict, Counter

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class LayerWiseProtoNet(nn.Module):
    def __init__(
        self,
        input_dim=4096,
        embed_dim=256,
        hidden_dim=1024,
        num_classes=2,
        num_layers=32,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.embed_dim = embed_dim

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
        if x.ndim != 3:
            raise ValueError(f"Expected x [B,L,H], got {tuple(x.shape)}")

        z = self.encoder(x)  # [B,L,D]
        z = F.normalize(z, dim=-1)

        p = F.normalize(self.prototypes, dim=-1)  # [L,C,D]

        sim = torch.einsum("bld,lcd->blc", z, p)  # [B,L,C]
        logits = sim.mean(dim=1)  # [B,C]

        return logits, sim, z, p


def layerwise_feature(x):
    """
    x: [L,T,H]
    return: [L,H]
    """
    x = torch.tensor(x, dtype=torch.float32)

    if x.ndim != 3:
        raise ValueError(f"Expected [L,T,H], got {tuple(x.shape)}")

    # target_word subword 평균만 수행. layer 평균은 하지 않음.
    x = x.mean(dim=1)  # [L,H]
    return x


def load_samples(h5_dir, split_name, allowed_concepts=("safe", "existence")):
    paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
    if not paths:
        raise FileNotFoundError(f"No h5 files found in {h5_dir} for split={split_name}")

    samples = []
    raw_counter = Counter()
    kept_counter = Counter()
    skip_counter = Counter()

    for path in paths:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]

                concept = str(g.attrs.get("concept", ""))
                ann = str(g.attrs.get("annotation_type", ""))
                answer_text = str(g.attrs.get("answer_text", "")).strip()
                target_word = str(g.attrs.get("target_word", "")).strip()

                raw_counter[(concept, ann)] += 1

                if ann != "generative":
                    skip_counter["not_generative"] += 1
                    continue

                if concept not in allowed_concepts:
                    skip_counter["concept_not_allowed"] += 1
                    continue

                if target_word == "":
                    skip_counter["empty_target_word"] += 1
                    continue

                if answer_text.lower() != target_word.lower():
                    skip_counter["answer_target_mismatch"] += 1
                    continue

                if "hidden_states" not in g:
                    skip_counter["missing_hidden_states"] += 1
                    continue

                samples.append({
                    "path": path,
                    "key": key,
                    "concept": concept,
                    "source": g.attrs.get("source", ""),
                    "annotation_type": ann,
                    "image": g.attrs.get("image", ""),
                    "query": g.attrs.get("query", ""),
                    "answer_text": answer_text,
                    "target_word": target_word,
                    "id": str(g.attrs.get("id", "")),
                })
                kept_counter[concept] += 1

    print("H5 files:", len(paths))
    print("Raw count:", raw_counter)
    print("Kept samples:", len(samples))
    print("Kept concept count:", kept_counter)
    print("Skipped:", skip_counter)

    if len(samples) == 0:
        raise RuntimeError("No usable samples found.")

    return samples


def load_feature(sample):
    with h5py.File(sample["path"], "r") as f:
        g = f[sample["key"]]
        x = g["hidden_states"][:]
    return layerwise_feature(x)


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    ckpt_args = ckpt.get("args", {})
    embed_dim = ckpt_args.get("embed_dim", args.embed_dim)
    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)
    num_layers = ckpt_args.get("num_layers", ckpt.get("num_layers", args.num_layers))

    print("Checkpoint:", args.checkpoint)
    print("Label map:", label_map)
    print("Num layers:", num_layers)

    model = LayerWiseProtoNet(
        input_dim=4096,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_map),
        num_layers=num_layers,
    )

    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    samples = load_samples(
        args.h5_dir,
        args.split_name,
        allowed_concepts=tuple(label_map.keys()),
    )

    confusion = defaultdict(lambda: defaultdict(int))
    total = 0
    correct = 0

    layer_sum = {
        concept: {
            "existence": [0.0 for _ in range(num_layers)],
            "safe": [0.0 for _ in range(num_layers)],
            "margin_existence_minus_safe": [0.0 for _ in range(num_layers)],
            "count": 0,
        }
        for concept in label_map.keys()
    }

    layer_pred_count = {
        concept: [Counter() for _ in range(num_layers)]
        for concept in label_map.keys()
    }

    top_by_layer = {
        layer: {
            concept: [] for concept in label_map.keys()
        }
        for layer in range(num_layers)
    }

    sample_outputs = []

    with torch.no_grad():
        for sample in tqdm(samples):
            x = load_feature(sample).unsqueeze(0).to(device)  # [1,L,H]

            logits, sim, z, p = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            sim = sim.squeeze(0).cpu()  # [L,C]

            pred_id = int(torch.argmax(probs).item())
            pred_concept = inv_label_map[pred_id]
            true_concept = sample["concept"]

            confusion[true_concept][pred_concept] += 1
            total += 1
            correct += int(pred_concept == true_concept)

            existence_id = label_map.get("existence", None)
            safe_id = label_map.get("safe", None)

            trajectory = []
            for layer in range(num_layers):
                layer_scores = {
                    inv_label_map[c]: float(sim[layer, c].item())
                    for c in range(sim.shape[-1])
                }

                layer_pred_id = int(torch.argmax(sim[layer]).item())
                layer_pred_concept = inv_label_map[layer_pred_id]
                layer_pred_count[true_concept][layer][layer_pred_concept] += 1

                ex_sim = layer_scores.get("existence", 0.0)
                safe_sim = layer_scores.get("safe", 0.0)
                margin = ex_sim - safe_sim

                layer_sum[true_concept]["existence"][layer] += ex_sim
                layer_sum[true_concept]["safe"][layer] += safe_sim
                layer_sum[true_concept]["margin_existence_minus_safe"][layer] += margin

                trajectory.append({
                    "layer": layer,
                    "existence_sim": ex_sim,
                    "safe_sim": safe_sim,
                    "margin_existence_minus_safe": margin,
                    "layer_pred": layer_pred_concept,
                })

                for proto_concept, sim_value in layer_scores.items():
                    top_by_layer[layer][proto_concept].append({
                        "similarity": sim_value,
                        "true_concept": true_concept,
                        "pred_concept": pred_concept,
                        "layer_pred": layer_pred_concept,
                        "target_word": sample["target_word"],
                        "answer_text": sample["answer_text"],
                        "image": sample["image"],
                        "id": sample["id"],
                        "key": sample["key"],
                        "h5_path": sample["path"],
                    })

            layer_sum[true_concept]["count"] += 1

            if len(sample_outputs) < args.num_sample_outputs:
                sample_outputs.append({
                    "true_concept": true_concept,
                    "pred_concept": pred_concept,
                    "prob": {
                        inv_label_map[i]: float(probs[i].item())
                        for i in range(len(inv_label_map))
                    },
                    "target_word": sample["target_word"],
                    "answer_text": sample["answer_text"],
                    "image": sample["image"],
                    "id": sample["id"],
                    "trajectory": trajectory,
                })

    # 평균 layer trajectory 계산
    avg_layer_trajectory = {}
    for concept, stat in layer_sum.items():
        count = max(stat["count"], 1)
        avg_layer_trajectory[concept] = {
            "count": stat["count"],
            "existence_sim": [v / count for v in stat["existence"]],
            "safe_sim": [v / count for v in stat["safe"]],
            "margin_existence_minus_safe": [
                v / count for v in stat["margin_existence_minus_safe"]
            ],
            "layer_pred_count": [
                dict(layer_pred_count[concept][layer])
                for layer in range(num_layers)
            ],
        }

    # layer별 nearest sample 저장
    nearest_by_layer = {}
    for layer in range(num_layers):
        nearest_by_layer[str(layer)] = {}
        for proto_concept, rows in top_by_layer[layer].items():
            rows = sorted(rows, key=lambda x: x["similarity"], reverse=True)
            nearest_by_layer[str(layer)][proto_concept] = rows[:args.top_k]

    summary = {
        "checkpoint": args.checkpoint,
        "h5_dir": args.h5_dir,
        "split_name": args.split_name,
        "label_map": label_map,
        "num_layers": num_layers,
        "accuracy_on_loaded_samples": correct / max(total, 1),
        "total_loaded_samples": total,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    output = {
        "summary": summary,
        "avg_layer_trajectory": avg_layer_trajectory,
        "sample_trajectories": sample_outputs,
        "nearest_samples_by_layer_prototype": nearest_by_layer,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "layerwise_prototype_inspection_existence_only.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nAccuracy on loaded samples:", summary["accuracy_on_loaded_samples"])

    print("\nConfusion:")
    for true_c, row in summary["confusion"].items():
        print(true_c, row)

    print("\nAverage layer trajectory")
    for concept, tr in avg_layer_trajectory.items():
        print(f"\n[{concept}] count={tr['count']}")
        for layer in range(num_layers):
            print(
                f"L{layer:02d} "
                f"exist={tr['existence_sim'][layer]:.4f} "
                f"safe={tr['safe_sim'][layer]:.4f} "
                f"margin={tr['margin_existence_minus_safe'][layer]:.4f} "
                f"pred_count={tr['layer_pred_count'][layer]}"
            )

    print("\nNearest samples by selected layers:")
    for layer in [0, num_layers // 2, num_layers - 1]:
        print(f"\nLayer {layer}")
        for proto_concept in label_map.keys():
            rows = nearest_by_layer[str(layer)][proto_concept]
            print(f"  [{proto_concept}]")
            for r in rows[:5]:
                print(
                    f"    sim={r['similarity']:.4f} "
                    f"true={r['true_concept']} pred={r['pred_concept']} "
                    f"layer_pred={r['layer_pred']} "
                    f"target={r['target_word']} image={r['image']}"
                )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoint/amber_layerwise_prototype_existence_generative/prototype_best.pt",
    )
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="generative_only")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/analysis/amber_layerwise_prototype_existence_generative",
    )

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-sample-outputs", type=int, default=20)

    parser.add_argument("--num-layers", type=int, default=32)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    args = parser.parse_args()
    main(args)