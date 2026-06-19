import argparse
import glob
import json
import os
from collections import defaultdict

import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


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


def pool_hidden(x, layer_mode="mean_last_k", last_k=4):
    x = torch.tensor(x, dtype=torch.float32)

    if x.ndim == 3:
        x = x.mean(dim=1)

    if layer_mode == "last":
        return x[-1]
    if layer_mode == "mean_last_k":
        return x[-last_k:].mean(dim=0)
    if layer_mode == "mean_all":
        return x.mean(dim=0)

    raise ValueError(layer_mode)


def load_samples(h5_dir, split_name):
    paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
    if not paths:
        raise FileNotFoundError(f"No h5 files found for split={split_name}")

    samples = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                samples.append({
                    "path": path,
                    "key": key,
                    "concept": g.attrs["concept"],
                    "source": g.attrs.get("source", ""),
                    "annotation_type": g.attrs.get("annotation_type", ""),
                    "image": g.attrs.get("image", ""),
                    "query": g.attrs.get("query", ""),
                    "answer_text": g.attrs.get("answer_text", ""),
                    "target_word": g.attrs.get("target_word", ""),
                })
    return samples


def load_feature(sample, use_query_for, layer_mode, last_k):
    with h5py.File(sample["path"], "r") as f:
        g = f[sample["key"]]
        if sample["concept"] in use_query_for:
            x = g["query_hidden_states"][:]
        else:
            x = g["hidden_states"][:]

    return pool_hidden(x, layer_mode=layer_mode, last_k=last_k)


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    label_map = ckpt["label_map"]
    inv_label_map = {v: k for k, v in label_map.items()}

    model_args = ckpt.get("prototype_args", {})
    cbm_args = ckpt.get("args", {})

    embed_dim = model_args.get("embed_dim", args.embed_dim)
    proto_hidden_dim = model_args.get("hidden_dim", args.proto_hidden_dim)
    cbm_hidden_dim = cbm_args.get("cbm_hidden_dim", args.cbm_hidden_dim)

    layer_mode = model_args.get("layer_mode", "mean_last_k")
    last_k = model_args.get("last_k", 4)

    proto = PrototypeEncoder(
        input_dim=4096,
        embed_dim=embed_dim,
        hidden_dim=proto_hidden_dim,
        num_classes=len(label_map),
    )
    cbm = ConceptBottleneck(
        num_concepts=len(label_map),
        hidden_dim=cbm_hidden_dim,
    )

    proto.load_state_dict(ckpt["prototype"])
    cbm.load_state_dict(ckpt["cbm"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proto.to(device).eval()
    cbm.to(device).eval()

    use_query_for = set(args.use_query_for.split(",")) if args.use_query_for else {"attribute", "relation"}
    samples = load_samples(args.h5_dir, args.split_name)

    confusion = defaultdict(lambda: defaultdict(int))
    concept_scores_by_true = defaultdict(list)
    top_examples = defaultdict(list)

    with torch.no_grad():
        for sample in tqdm(samples):
            x = load_feature(sample, use_query_for, layer_mode, last_k).unsqueeze(0).to(device)

            proto_sim, z, p = proto(x)
            logits, score = cbm(proto_sim)

            prob = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            score = score.squeeze(0).cpu()
            proto_sim = proto_sim.squeeze(0).cpu()

            pred_id = int(prob.argmax().item())
            pred_concept = inv_label_map[pred_id]
            true_concept = sample["concept"]

            confusion[true_concept][pred_concept] += 1
            concept_scores_by_true[true_concept].append(score.tolist())

            row = {
                "true_concept": true_concept,
                "pred_concept": pred_concept,
                "pred_prob": float(prob[pred_id]),
                "scores": {inv_label_map[i]: float(score[i]) for i in range(len(label_map))},
                "proto_similarity": {inv_label_map[i]: float(proto_sim[i]) for i in range(len(label_map))},
                "key": sample["key"],
                "image": sample["image"],
                "source": sample["source"],
                "annotation_type": sample["annotation_type"],
                "query": sample["query"],
                "answer_text": sample["answer_text"],
                "target_word": sample["target_word"],
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

    for c in list(top_examples.keys()):
        top_examples[c] = sorted(top_examples[c], key=lambda r: r["pred_prob"], reverse=True)[:args.top_k]

    output = {
        "summary": {
            "checkpoint": args.checkpoint,
            "h5_dir": args.h5_dir,
            "split_name": args.split_name,
            "label_map": label_map,
            "layer_mode": layer_mode,
            "last_k": last_k,
            "confusion": {k: dict(v) for k, v in confusion.items()},
            "avg_concept_scores_by_true": avg_scores,
        },
        "top_examples_by_pred": top_examples,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "cbm_inspection.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nConfusion:")
    for true_c, row in output["summary"]["confusion"].items():
        print(true_c, row)

    print("\nAverage CBM concept scores by true concept:")
    for true_c, row in avg_scores.items():
        print(true_c, row)

    print("\nTop examples:")
    for concept, rows in top_examples.items():
        print(f"\n[{concept}]")
        for r in rows[:5]:
            print(
                f"prob={r['pred_prob']:.4f} "
                f"true={r['true_concept']} pred={r['pred_concept']} "
                f"image={r['image']} answer={r['answer_text']} "
                f"query={r['query'][:80]}"
            )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="output/checkpoint/amber_cbm_full/cbm_best.pt")
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="full")
    parser.add_argument("--out-dir", type=str, default="output/analysis/amber_cbm_full")

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-query-for", type=str, default="attribute,relation")

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--proto-hidden-dim", type=int, default=1024)
    parser.add_argument("--cbm-hidden-dim", type=int, default=64)

    args = parser.parse_args()
    main(args)