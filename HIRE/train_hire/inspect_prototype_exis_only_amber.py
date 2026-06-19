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


class ProtoNet(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=2):
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

    if x.ndim != 3:
        raise ValueError(f"Expected [L,T,H], got {tuple(x.shape)}")

    # target_word가 subword 여러 개면 token 평균
    x = x.mean(dim=1)  # [L,H]

    if layer_mode == "last":
        return x[-1]
    if layer_mode == "mean_last_k":
        return x[-last_k:].mean(dim=0)
    if layer_mode == "mean_all":
        return x.mean(dim=0)

    raise ValueError(layer_mode)


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
                    "id": g.attrs.get("id", ""),
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


def load_feature(sample, layer_mode, last_k):
    with h5py.File(sample["path"], "r") as f:
        g = f[sample["key"]]
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

    print("Checkpoint:", args.checkpoint)
    print("Label map:", label_map)
    print("Layer mode:", layer_mode)
    print("Last k:", last_k)

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

    samples = load_samples(
        args.h5_dir,
        args.split_name,
        allowed_concepts=tuple(label_map.keys()),
    )

    results = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))
    per_word = defaultdict(lambda: defaultdict(int))

    total = 0
    correct = 0

    with torch.no_grad():
        _, _, prototypes = model(torch.zeros(1, 4096, device=device))
        prototypes = prototypes.cpu()

        for sample in tqdm(samples):
            x = load_feature(sample, layer_mode, last_k).unsqueeze(0).to(device)

            logits, z, _ = model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()
            sims = (z.cpu() @ prototypes.t()).squeeze(0)

            pred_id = int(torch.argmax(probs).item())
            pred_concept = inv_label_map[pred_id]
            true_concept = sample["concept"]

            confusion[true_concept][pred_concept] += 1
            per_word[sample["target_word"]][true_concept] += 1

            total += 1
            correct += int(pred_concept == true_concept)

            for proto_id, sim in enumerate(sims.tolist()):
                proto_concept = inv_label_map[proto_id]
                results[proto_concept].append({
                    "similarity": sim,
                    "true_concept": true_concept,
                    "pred_concept": pred_concept,
                    "prob": float(probs[proto_id]),
                    "key": sample["key"],
                    "id": str(sample["id"]),
                    "image": sample["image"],
                    "source": sample["source"],
                    "annotation_type": sample["annotation_type"],
                    "query": sample["query"],
                    "answer_text": sample["answer_text"],
                    "target_word": sample["target_word"],
                    "h5_path": sample["path"],
                })

    os.makedirs(args.out_dir, exist_ok=True)

    top_results = {}
    for concept, rows in results.items():
        rows = sorted(rows, key=lambda x: x["similarity"], reverse=True)
        top_results[concept] = rows[:args.top_k]

    summary = {
        "checkpoint": args.checkpoint,
        "h5_dir": args.h5_dir,
        "split_name": args.split_name,
        "layer_mode": layer_mode,
        "last_k": last_k,
        "label_map": label_map,
        "accuracy_on_loaded_samples": correct / max(total, 1),
        "total_loaded_samples": total,
        "confusion": {k: dict(v) for k, v in confusion.items()},
    }

    output = {
        "summary": summary,
        "nearest_samples_by_prototype": top_results,
    }

    out_path = os.path.join(args.out_dir, "prototype_inspection_existence_only.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nAccuracy on loaded samples:", summary["accuracy_on_loaded_samples"])

    print("\nConfusion:")
    for true_c, row in summary["confusion"].items():
        print(true_c, row)

    print("\nNearest samples:")
    for concept, rows in top_results.items():
        print(f"\n[{concept}]")
        for r in rows[:10]:
            print(
                f"sim={r['similarity']:.4f} "
                f"true={r['true_concept']} pred={r['pred_concept']} "
                f"prob={r['prob']:.4f} "
                f"target={r['target_word']} "
                f"image={r['image']}"
            )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="output/checkpoint/amber_prototype_existence_generative/prototype_best.pt",
    )
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="generative_only")
    parser.add_argument("--out-dir", type=str, default="output/analysis/amber_prototype_existence_generative")

    parser.add_argument("--top-k", type=int, default=30)

    parser.add_argument("--layer-mode", type=str, default=None, choices=[None, "last", "mean_last_k", "mean_all"])
    parser.add_argument("--last-k", type=int, default=None)

    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=1024)

    args = parser.parse_args()
    main(args)