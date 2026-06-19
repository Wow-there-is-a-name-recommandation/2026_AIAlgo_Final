import argparse, glob, json, os
from collections import Counter

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class LayerWisePrototypeEncoder(nn.Module):
    def __init__(self, input_dim=4096, embed_dim=256, hidden_dim=1024, num_classes=2, num_layers=32):
        super().__init__()
        self.num_layers = num_layers
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
        return sim


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_summary_table(proto_json, chair_json, out_dir):
    proto = load_json(proto_json)
    chair = load_json(chair_json)

    summary = proto["summary"]
    traj = proto["avg_layer_trajectory"]
    chair_summary = chair["summary"]
    corr = chair["correlations"]

    safe_margin = np.array(traj["safe"]["margin_existence_minus_safe"])
    exist_margin = np.array(traj["existence"]["margin_existence_minus_safe"])

    rows = [
        ["Metric", "Value"],
        ["Prototype accuracy", f"{summary['accuracy_on_loaded_samples'] * 100:.2f}%"],
        ["Total AMBER samples", f"{summary['total_loaded_samples']}"],
        ["Safe samples", f"{traj['safe']['count']}"],
        ["Existence samples", f"{traj['existence']['count']}"],
        ["Safe mean margin", f"{safe_margin.mean():.3f}"],
        ["Existence mean margin", f"{exist_margin.mean():.3f}"],
        ["Safe L2-L31 margin", f"{safe_margin[2:].mean():.3f}"],
        ["Existence L2-L31 margin", f"{exist_margin[2:].mean():.3f}"],
        ["COCO matched samples", f"{chair_summary['num_matched']}"],
        ["Avg CHAIRi", f"{chair_summary['avg_CHAIRi']:.3f}"],
        ["Avg CHAIRs", f"{chair_summary['avg_CHAIRs']:.3f}"],
        ["CHAIRi corr. top5 exist", f"{corr['CHAIRi']['top5_existence_score']:.3f}"],
        ["CHAIRs corr. top5 exist", f"{corr['CHAIRs']['top5_existence_score']:.3f}"],
    ]

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    ax.axis("off")

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.28)

    plt.title("Prototype Separation and CHAIR Correlation Summary")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "prototype_chair_summary_table.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)


def load_samples(h5_dir, split_name, label_map):
    paths = sorted(glob.glob(os.path.join(h5_dir, f"hs_amber_proto_{split_name}_*_0.h5")))
    samples = []

    for path in paths:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                g = f[key]
                concept = str(g.attrs.get("concept", ""))
                ann = str(g.attrs.get("annotation_type", ""))
                answer = str(g.attrs.get("answer_text", "")).strip().lower()
                target = str(g.attrs.get("target_word", "")).strip().lower()

                if ann != "generative":
                    continue
                if concept not in label_map:
                    continue
                if not target:
                    continue
                if answer != target:
                    continue
                if "hidden_states" not in g:
                    continue

                samples.append((path, key, concept))

    return samples


def read_feature(path, key):
    with h5py.File(path, "r") as f:
        x = f[key]["hidden_states"][:]
    x = torch.tensor(x, dtype=torch.float32)
    x = x.mean(dim=1)  # [L,H]
    return x


def parse_layer_groups(group_str):
    """
    예:
    "early:0,1,2,3;middle:14,15,16,17;late:28,29,30,31"
    """
    groups = {}

    for part in group_str.split(";"):
        name, layers = part.split(":")
        groups[name] = [int(x) for x in layers.split(",") if x.strip() != ""]

    return groups


def compute_group_margins(args):
    ckpt = torch.load(args.prototype_checkpoint, map_location="cpu")
    label_map = ckpt["label_map"]
    proto_args = ckpt.get("args", {})
    num_layers = ckpt.get("num_layers", proto_args.get("num_layers", 32))

    model = LayerWisePrototypeEncoder(
        input_dim=4096,
        embed_dim=proto_args.get("embed_dim", 256),
        hidden_dim=proto_args.get("hidden_dim", 1024),
        num_classes=len(label_map),
        num_layers=num_layers,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    existence_id = label_map["existence"]
    safe_id = label_map["safe"]

    samples = load_samples(args.h5_dir, args.split_name, label_map)
    groups = parse_layer_groups(args.layer_groups)

    margins = {
        group_name: {"safe": [], "existence": []}
        for group_name in groups
    }

    with torch.no_grad():
        for path, key, concept in samples:
            x = read_feature(path, key).unsqueeze(0).to(device)
            sim = model(x).squeeze(0).cpu()  # [L,C]

            margin_per_layer = sim[:, existence_id] - sim[:, safe_id]  # [L]

            for group_name, layer_ids in groups.items():
                group_margin = float(margin_per_layer[layer_ids].mean().item())
                margins[group_name][concept].append(group_margin)

    return margins, groups


def save_group_margin_histograms(args, out_dir):
    margins, groups = compute_group_margins(args)

    bins = np.linspace(-2.2, 2.2, 50)

    for group_name, group_data in margins.items():
        plt.figure(figsize=(8, 4.8))

        plt.hist(
            group_data["safe"],
            bins=bins,
            alpha=0.65,
            density=True,
            label="Safe",
        )
        plt.hist(
            group_data["existence"],
            bins=bins,
            alpha=0.65,
            density=True,
            label="Existence",
        )

        plt.axvline(0, linestyle="--", linewidth=1)
        plt.xlabel("Mean margin: existence similarity - safe similarity")
        plt.ylabel("Density")

        layer_text = ",".join(str(x) for x in groups[group_name])
        plt.title(f"Prototype Margin Distribution ({group_name}: layers {layer_text})")

        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"margin_distribution_{group_name}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)


def save_margin_line(proto_json, out_dir):
    proto = load_json(proto_json)
    traj = proto["avg_layer_trajectory"]

    safe = np.array(traj["safe"]["margin_existence_minus_safe"])
    exist = np.array(traj["existence"]["margin_existence_minus_safe"])
    layers = np.arange(len(safe))

    plt.figure(figsize=(9, 4.8))
    plt.plot(layers, safe, marker="o", label="Safe")
    plt.plot(layers, exist, marker="o", label="Existence")
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Transformer layer")
    plt.ylabel("Existence sim - Safe sim")
    plt.title("Layer-wise Prototype Margin")
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, "layerwise_margin_line.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved:", out_path)

def save_layer_prototype_similarity_heatmaps(args, out_dir):
    ckpt = torch.load(args.prototype_checkpoint, map_location="cpu")
    label_map = ckpt["label_map"]
    proto_args = ckpt.get("args", {})
    num_layers = ckpt.get("num_layers", proto_args.get("num_layers", 32))

    model = LayerWisePrototypeEncoder(
        input_dim=4096,
        embed_dim=proto_args.get("embed_dim", 256),
        hidden_dim=proto_args.get("hidden_dim", 1024),
        num_classes=len(label_map),
        num_layers=num_layers,
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    with torch.no_grad():
        p = F.normalize(model.prototypes, dim=-1).cpu()  # [L,C,D]

    existence_id = label_map["existence"]
    safe_id = label_map["safe"]

    exist_proto = p[:, existence_id, :]  # [L,D]
    safe_proto = p[:, safe_id, :]        # [L,D]

    exist_layer_sim = exist_proto @ exist_proto.t()
    safe_layer_sim = safe_proto @ safe_proto.t()
    cross_layer_sim = exist_proto @ safe_proto.t()

    def plot_heatmap(mat, title, filename, vmin=-1.0, vmax=1.0):
        mat = mat.numpy()

        plt.figure(figsize=(7, 6))
        im = plt.imshow(mat, cmap="coolwarm", vmin=vmin, vmax=vmax, aspect="auto")

        plt.xlabel("Layer")
        plt.ylabel("Layer")
        plt.title(title)

        ticks = np.arange(num_layers)
        plt.xticks(ticks, ticks, fontsize=7)
        plt.yticks(ticks, ticks, fontsize=7)

        cbar = plt.colorbar(im)
        cbar.set_label("Cosine similarity")

        plt.tight_layout()
        out_path = os.path.join(out_dir, filename)
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("Saved:", out_path)

    plot_heatmap(
        exist_layer_sim,
        "Existence Prototype Layer Similarity",
        "existence_prototype_layer_similarity_heatmap.png",
    )

    plot_heatmap(
        safe_layer_sim,
        "Safe Prototype Layer Similarity",
        "safe_prototype_layer_similarity_heatmap.png",
    )

    plot_heatmap(
        cross_layer_sim,
        "Existence-to-Safe Prototype Cross-layer Similarity",
        "existence_safe_cross_layer_similarity_heatmap.png",
    )


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    save_summary_table(args.prototype_inspection, args.chair_correlation, args.out_dir)
    save_margin_line(args.prototype_inspection, args.out_dir)
    save_group_margin_histograms(args, args.out_dir)
    save_layer_prototype_similarity_heatmaps(args, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prototype-inspection",
        type=str,
        default="output/analysis/amber_layerwise_prototype_existence_generative/layerwise_prototype_inspection_existence_only.json",
    )
    parser.add_argument(
        "--chair-correlation",
        type=str,
        default="output/analysis/layerwise_proto_activation_chair/exis_only_cbm_chair_correlation.json",
    )
    parser.add_argument(
        "--prototype-checkpoint",
        type=str,
        default="output/checkpoint/amber_layerwise_prototype_existence_generative/prototype_best.pt",
    )
    parser.add_argument(
        "--layer-groups",
        type=str,
        default="early:0,1,2,3;middle:14,15,16,17;late:28,29,30,31",
    )
    parser.add_argument("--h5-dir", type=str, default="output/hidden_states_amber_proto_shard")
    parser.add_argument("--split-name", type=str, default="generative_only")
    parser.add_argument("--hist-layer", type=int, default=10)
    parser.add_argument("--out-dir", type=str, default="output/figures/layerwise_prototype_ppt")

    args = parser.parse_args()
    main(args)