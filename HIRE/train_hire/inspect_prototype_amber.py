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


class ProtoNet(nn.Module):
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
        z = self.encoder(x)
        z = F.normalize(z, dim=-1)
        p = F.normalize(self.prototypes, dim=-1)
        logits = z @ p.t()
        return logits, z, p


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
        raise FileNotFoundError(f"No h5 files found in {h5_dir} for split={split_name}")

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

    ckpt_args = ckpt.get("args", {})
    embed_dim = ckpt_args.get("embed_dim", args.embed_dim)
    hidden_dim = ckpt_args.get("hidden_dim", args.hidden_dim)
    layer_mode = args.layer_mode or ckpt_args.get("layer_mode", "mean_last_k")
    last_k = args.last_k or ckpt_args.get("last_k", 4)

    model = ProtoNet(
        input_dim=4096,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=len(label_map),
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    use_query_for = set(args.use_query_for.split(",")) if args.use_query_for else {"attribute", "relation"}

    samples = load_samples(args.h5_dir, args.split_name)

    results = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))

    with torch.no_grad():
        _, _, prototypes = model(torch.zeros(1, 4096, device=device))
        prototypes = prototypes.cpu()

        for sample in tqdm(samples):
            x = load_feature(sample, use_query_for, layer_mode, last_k).unsqueeze(0).to(device)

            logits, z, p = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            sims = (z.cpu() @ prototypes.t()).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            pred_concept = inv_label_map[pred_id]

            true_concept = sample["concept"]
            confusion[true_concept][pred_concept] += 1

            for proto_id, sim in enumerate(sims.tolist()):
                proto_concept = inv_label_map[proto_id]
                results[proto_concept].append({
                    "similarity": sim,
                    "true_concept": true_concept,
                    "pred_concept": pred_concept,
                    "prob": float(probs[proto_id]),
                    "key": sample["key"],
                    "image": sample["image"],
                    "source": sample["source"],
                    "annotation_type": sample["annotation_type"],
                    "query": sample["query"],
                    "answer_text": sample["answer_text"],
                    "target_word": sample["target_word"],
                    "h5_path": sample["path"],
                })

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {
        "checkpoint": args.checkpoint,
        "h5_dir": args.h5_dir,
        "split_name": args.split_name,
        "layer_mode": layer_mode,
        "last_k": last_k,
        "label_map": label_map,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    for concept, rows in results.items():
        rows = sorted(rows, key=lambda x: x["similarity"], reverse=True)
        results[concept] = rows[:args.top_k]

    output = {
        "summary": summary,
        "nearest_samples_by_prototype": results,
    }

    out_path = os.path.join(args.out_dir, "prototype_inspection.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nConfusion:")
    for true_c, row in summary["confusion"].items():
        print(true_c, row)

    print("\nNearest samples:")
    for concept, rows in results.items():
        print(f"\n[{concept}]")
        for r in rows[:5]:
            print(
                f"sim={r['similarity']:.4f} "
                f"true={r['true_concept']} pred={r['pred_concept']} "
                f"image={r['image']} answer={r['answer_text']} "
                f"query={r['query'][:80]}"
            )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", type=str, default="output/checkpoint/amber_prototype_full/prototype_best.pt")
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="full")
    parser.add_argument("--out-dir", type=str, default="output/analysis/amber_prototype_full")

    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--use-query-for", type=str, default="attribute,relation")

    parser.add_argument("--layer-mode", type=str, default=None, choices=[None, "last", "mean_last_k", "mean_all"])
    parser.add_argument("--last-k", type=int, default=None)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    args = parser.parse_args()
    main(args)