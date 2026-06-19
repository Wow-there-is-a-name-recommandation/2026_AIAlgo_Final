import argparse, glob, json, os
from collections import defaultdict, Counter

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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
        z = F.normalize(self.encoder(x), dim=-1)          # [B,L,D]
        p = F.normalize(self.prototypes, dim=-1)          # [L,C,D]
        sim = torch.einsum("bld,lcd->blc", z, p)          # [B,L,C]
        logits = sim.mean(dim=1)                          # [B,C]
        return logits, sim, z, p


class LayerWiseConceptBottleneck(nn.Module):
    def __init__(self, num_layers=32, num_concepts=2, hidden_dim=64):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(num_layers * num_concepts, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
        )

    def forward(self, sim):
        x = sim.flatten(start_dim=1)       # [B, L*C]
        logits = self.bottleneck(x)        # [B,C]
        score = torch.sigmoid(logits)
        return logits, score


def to_layerwise_feature(x):
    x = torch.tensor(x, dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"Expected [L,T,H], got {tuple(x.shape)}")
    return x.mean(dim=1)  # [L,H]


def load_samples(h5_dir, split_name, allowed_concepts):
    paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
    if not paths:
        raise FileNotFoundError(f"No h5 files found for split={split_name}")

    samples = []
    raw_counter, kept_counter, skip_counter = Counter(), Counter(), Counter()

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
                    "image": g.attrs.get("image", ""),
                    "query": g.attrs.get("query", ""),
                    "answer_text": answer_text,
                    "target_word": target_word,
                    "annotation_type": ann,
                    "source": g.attrs.get("source", ""),
                    "id": str(g.attrs.get("id", "")),
                })
                kept_counter[concept] += 1

    print("H5 files:", len(paths))
    print("Raw count:", raw_counter)
    print("Kept samples:", len(samples))
    print("Kept concept count:", kept_counter)
    print("Skipped:", skip_counter)

    return samples


def load_feature(sample):
    with h5py.File(sample["path"], "r") as f:
        x = f[sample["key"]]["hidden_states"][:]
    return to_layerwise_feature(x)


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    proto_args = ckpt.get("prototype_args", {})
    cbm_args = ckpt.get("args", {})
    num_layers = ckpt.get("num_layers", proto_args.get("num_layers", args.num_layers))

    embed_dim = proto_args.get("embed_dim", args.embed_dim)
    proto_hidden_dim = proto_args.get("hidden_dim", args.proto_hidden_dim)
    cbm_hidden_dim = cbm_args.get("cbm_hidden_dim", args.cbm_hidden_dim)

    print("Checkpoint:", args.checkpoint)
    print("Label map:", label_map)
    print("Num layers:", num_layers)

    proto = LayerWisePrototypeEncoder(
        input_dim=4096,
        embed_dim=embed_dim,
        hidden_dim=proto_hidden_dim,
        num_classes=len(label_map),
        num_layers=num_layers,
    )

    cbm = LayerWiseConceptBottleneck(
        num_layers=num_layers,
        num_concepts=len(label_map),
        hidden_dim=cbm_hidden_dim,
    )

    proto.load_state_dict(ckpt["prototype"])
    cbm.load_state_dict(ckpt["cbm"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto.to(device).eval()
    cbm.to(device).eval()

    samples = load_samples(args.h5_dir, args.split_name, allowed_concepts=tuple(label_map.keys()))

    confusion = defaultdict(lambda: defaultdict(int))
    concept_scores_by_true = defaultdict(list)
    layer_sim_by_true = defaultdict(lambda: {
        "existence": [0.0 for _ in range(num_layers)],
        "safe": [0.0 for _ in range(num_layers)],
        "margin": [0.0 for _ in range(num_layers)],
        "count": 0,
    })
    top_examples = defaultdict(list)

    total, correct = 0, 0

    with torch.no_grad():
        for sample in tqdm(samples):
            x = load_feature(sample).unsqueeze(0).to(device)  # [1,L,H]

            _, sim, _, _ = proto(x)       # [1,L,C]
            logits, score = cbm(sim)      # [1,C]

            prob = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            score = score.squeeze(0).cpu()
            sim_cpu = sim.squeeze(0).cpu()

            pred_id = int(prob.argmax().item())
            pred_concept = inv_label_map[pred_id]
            true_concept = sample["concept"]

            confusion[true_concept][pred_concept] += 1
            total += 1
            correct += int(pred_concept == true_concept)
            concept_scores_by_true[true_concept].append(score.tolist())

            ex_id = label_map.get("existence")
            safe_id = label_map.get("safe")

            for layer in range(num_layers):
                ex_sim = float(sim_cpu[layer, ex_id]) if ex_id is not None else 0.0
                safe_sim = float(sim_cpu[layer, safe_id]) if safe_id is not None else 0.0
                layer_sim_by_true[true_concept]["existence"][layer] += ex_sim
                layer_sim_by_true[true_concept]["safe"][layer] += safe_sim
                layer_sim_by_true[true_concept]["margin"][layer] += ex_sim - safe_sim

            layer_sim_by_true[true_concept]["count"] += 1

            row = {
                "true_concept": true_concept,
                "pred_concept": pred_concept,
                "pred_prob": float(prob[pred_id]),
                "scores": {inv_label_map[i]: float(score[i]) for i in range(len(label_map))},
                "avg_layer_proto_similarity": {
                    inv_label_map[i]: float(sim_cpu[:, i].mean()) for i in range(len(label_map))
                },
                "target_word": sample["target_word"],
                "answer_text": sample["answer_text"],
                "image": sample["image"],
                "query": sample["query"],
                "key": sample["key"],
                "h5_path": sample["path"],
            }
            top_examples[pred_concept].append(row)

    avg_scores = {}
    for true_c, scores in concept_scores_by_true.items():
        t = torch.tensor(scores)
        avg_scores[true_c] = {
            inv_label_map[i]: float(t[:, i].mean().item())
            for i in range(t.shape[1])
        }

    avg_layer_sim = {}
    for true_c, stat in layer_sim_by_true.items():
        count = max(stat["count"], 1)
        avg_layer_sim[true_c] = {
            "count": stat["count"],
            "existence_sim": [v / count for v in stat["existence"]],
            "safe_sim": [v / count for v in stat["safe"]],
            "margin_existence_minus_safe": [v / count for v in stat["margin"]],
        }

    for c in list(top_examples.keys()):
        top_examples[c] = sorted(top_examples[c], key=lambda r: r["pred_prob"], reverse=True)[:args.top_k]

    output = {
        "summary": {
            "checkpoint": args.checkpoint,
            "h5_dir": args.h5_dir,
            "split_name": args.split_name,
            "label_map": label_map,
            "num_layers": num_layers,
            "accuracy_on_loaded_samples": correct / max(total, 1),
            "total_loaded_samples": total,
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "avg_concept_scores_by_true": avg_scores,
        },
        "avg_layer_proto_similarity_by_true": avg_layer_sim,
        "top_examples_by_pred": top_examples,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "layerwise_cbm_inspection.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nAccuracy:", output["summary"]["accuracy_on_loaded_samples"])

    print("\nConfusion:")
    for true_c, row in output["summary"]["confusion"].items():
        print(true_c, row)

    print("\nAverage CBM concept scores by true concept:")
    for true_c, row in avg_scores.items():
        print(true_c, row)

    print("\nAverage layer margin by true concept:")
    for true_c, row in avg_layer_sim.items():
        print(f"\n[{true_c}] count={row['count']}")
        for i, m in enumerate(row["margin_existence_minus_safe"]):
            print(f"L{i:02d} margin={m:.4f}")

    print("\nTop examples:")
    for concept, rows in top_examples.items():
        print(f"\n[{concept}]")
        for r in rows[:5]:
            print(
                f"prob={r['pred_prob']:.4f} "
                f"true={r['true_concept']} pred={r['pred_concept']} "
                f"target={r['target_word']} image={r['image']}"
            )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoint/amber_layerwise_cbm_existence_generative/cbm_best.pt",
    )
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="generative_only")
    parser.add_argument("--out-dir", type=str, default="output/analysis/amber_layerwise_cbm_existence_generative")

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--num-layers", type=int, default=32)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proto-hidden-dim", type=int, default=1024)
    parser.add_argument("--cbm-hidden-dim", type=int, default=64)

    args = parser.parse_args()
    main(args)