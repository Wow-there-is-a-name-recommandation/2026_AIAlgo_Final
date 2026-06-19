import json
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_result(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_margin_line_plot(data, out_dir):
    traj = data["avg_layer_trajectory"]

    layers = np.arange(len(traj["safe"]["margin_existence_minus_safe"]))

    safe_margin = np.array(traj["safe"]["margin_existence_minus_safe"])
    exist_margin = np.array(traj["existence"]["margin_existence_minus_safe"])

    plt.figure(figsize=(10, 4.8))
    plt.plot(layers, safe_margin, marker="o", label="Safe")
    plt.plot(layers, exist_margin, marker="o", label="Existence")
    plt.axhline(0, linestyle="--", linewidth=1)

    plt.xlabel("Transformer layer")
    plt.ylabel("Existence similarity - Safe similarity")
    plt.title("Layer-wise Prototype Margin")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "layerwise_margin_line.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def save_margin_heatmap(data, out_dir):
    traj = data["avg_layer_trajectory"]

    safe_margin = np.array(traj["safe"]["margin_existence_minus_safe"])
    exist_margin = np.array(traj["existence"]["margin_existence_minus_safe"])

    mat = np.stack([safe_margin, exist_margin], axis=0)

    plt.figure(figsize=(11, 2.8))
    im = plt.imshow(mat, aspect="auto", cmap="coolwarm", vmin=-2.0, vmax=2.0)

    plt.yticks([0, 1], ["Safe", "Existence"])
    plt.xticks(np.arange(mat.shape[1]), [str(i) for i in range(mat.shape[1])], fontsize=8)
    plt.xlabel("Transformer layer")
    plt.title("Layer-wise Prototype Margin Heatmap")

    cbar = plt.colorbar(im)
    cbar.set_label("Existence sim - Safe sim")

    plt.tight_layout()

    path = os.path.join(out_dir, "layerwise_margin_heatmap.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def save_summary_table(data, out_dir):
    summary = data["summary"]
    traj = data["avg_layer_trajectory"]

    safe_margin = np.array(traj["safe"]["margin_existence_minus_safe"])
    exist_margin = np.array(traj["existence"]["margin_existence_minus_safe"])

    rows = []
    rows.append(["Metric", "Value"])
    rows.append(["Accuracy", f"{summary['accuracy_on_loaded_samples']:.4f}"])
    rows.append(["Total samples", str(summary["total_loaded_samples"])])
    rows.append(["Safe count", str(traj["safe"]["count"])])
    rows.append(["Existence count", str(traj["existence"]["count"])])
    rows.append(["Safe mean margin", f"{safe_margin.mean():.4f}"])
    rows.append(["Existence mean margin", f"{exist_margin.mean():.4f}"])
    rows.append(["Safe L0 margin", f"{safe_margin[0]:.4f}"])
    rows.append(["Existence L0 margin", f"{exist_margin[0]:.4f}"])
    rows.append(["Safe L2-L31 mean margin", f"{safe_margin[2:].mean():.4f}"])
    rows.append(["Existence L2-L31 mean margin", f"{exist_margin[2:].mean():.4f}"])

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    plt.title("Layer-wise Prototype Summary")
    plt.tight_layout()

    path = os.path.join(out_dir, "layerwise_prototype_summary_table.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print("Saved:", path)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_result(args.input)

    save_margin_line_plot(data, args.out_dir)
    save_margin_heatmap(data, args.out_dir)
    save_summary_table(data, args.out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default="output/analysis/amber_layerwise_prototype_existence_generative/layerwise_prototype_inspection_existence_only.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="output/figures/layerwise_prototype",
    )

    args = parser.parse_args()
    main(args)