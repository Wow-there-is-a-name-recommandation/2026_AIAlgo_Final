import argparse
import json
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def collect_stats(trace, filter_action=None):
    pre_exis, post_exis, delta_exis = [], [], []
    pre_safe, post_safe, delta_safe = [], [], []
    pre_exis_prob, post_exis_prob, delta_exis_prob = [], [], []

    pre_margin, post_margin, delta_margin = [], [], []

    pre_layer_margin = []
    post_layer_margin = []
    delta_layer_margin = []

    router_actions = []

    for x in trace:
        action = x.get("router_action")

        if filter_action is not None and action != filter_action:
            continue

        px = x.get("pre_existence_score", 0.0)
        qx = x.get("post_existence_score", 0.0)
        ps = x.get("pre_safe_score", 0.0)
        qs = x.get("post_safe_score", 0.0)

        pre_exis.append(px)
        post_exis.append(qx)
        delta_exis.append(x.get("delta_existence_score", qx - px))

        pre_safe.append(ps)
        post_safe.append(qs)
        delta_safe.append(x.get("delta_safe_score", qs - ps))

        pre_exis_prob.append(x.get("pre_existence_prob", 0.0))
        post_exis_prob.append(x.get("post_existence_prob", 0.0))
        delta_exis_prob.append(x.get("delta_existence_prob", 0.0))

        pm = ps - px
        qm = qs - qx
        pre_margin.append(pm)
        post_margin.append(qm)
        delta_margin.append(qm - pm)

        pre_ex_layer = x.get("pre_existence_layer_proto_sim", [])
        post_ex_layer = x.get("post_existence_layer_proto_sim", [])
        pre_safe_layer = x.get("pre_safe_layer_proto_sim", [])
        post_safe_layer = x.get("post_safe_layer_proto_sim", [])

        if pre_ex_layer and post_ex_layer and pre_safe_layer and post_safe_layer:
            pre_m = [s - e for s, e in zip(pre_safe_layer, pre_ex_layer)]
            post_m = [s - e for s, e in zip(post_safe_layer, post_ex_layer)]
            delta_m = [b - a for a, b in zip(pre_m, post_m)]

            pre_layer_margin.append(pre_m)
            post_layer_margin.append(post_m)
            delta_layer_margin.append(delta_m)

        if action is not None:
            router_actions.append(action)

    return {
        "num_steps": len(pre_exis),

        "pre_existence_score": safe_mean(pre_exis),
        "post_existence_score": safe_mean(post_exis),
        "delta_existence_score": safe_mean(delta_exis),

        "pre_safe_score": safe_mean(pre_safe),
        "post_safe_score": safe_mean(post_safe),
        "delta_safe_score": safe_mean(delta_safe),

        "pre_existence_prob": safe_mean(pre_exis_prob),
        "post_existence_prob": safe_mean(post_exis_prob),
        "delta_existence_prob": safe_mean(delta_exis_prob),

        "pre_margin_safe_minus_existence": safe_mean(pre_margin),
        "post_margin_safe_minus_existence": safe_mean(post_margin),
        "delta_margin_safe_minus_existence": safe_mean(delta_margin),

        "router_edit_ratio": safe_mean(router_actions) if router_actions else 0.0,

        "_pre_layer_margin": pre_layer_margin,
        "_post_layer_margin": post_layer_margin,
        "_delta_layer_margin": delta_layer_margin,
    }


def average_layer_values(layer_rows):
    if not layer_rows:
        return []

    n_layers = len(layer_rows[0])
    avg = []
    for i in range(n_layers):
        vals = [row[i] for row in layer_rows if len(row) > i]
        avg.append(safe_mean(vals))
    return avg


def save_layer_csv(path, pre_avg, post_avg, delta_avg):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "layer",
                "pre_margin_safe_minus_existence",
                "post_margin_safe_minus_existence",
                "delta_margin_safe_minus_existence",
            ],
        )
        writer.writeheader()

        for i in range(len(delta_avg)):
            writer.writerow({
                "layer": i,
                "pre_margin_safe_minus_existence": pre_avg[i],
                "post_margin_safe_minus_existence": post_avg[i],
                "delta_margin_safe_minus_existence": delta_avg[i],
            })


def save_margin_heatmap(path, delta_avg, title):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if not delta_avg:
        print(f"No layer-wise margin data. Skip heatmap: {path}")
        return

    fig, ax = plt.subplots(figsize=(12, 2.2))
    im = ax.imshow([delta_avg], aspect="auto")

    ax.set_yticks([0])
    ax.set_yticklabels(["Δ margin"])
    ax.set_xticks(list(range(len(delta_avg))))
    ax.set_xticklabels(list(range(len(delta_avg))), rotation=90)
    ax.set_xlabel("Layer")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("post margin - pre margin")

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def strip_private_keys(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", default="output/dual_cbm_hire_analysis")
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--prefix", default="dual_cbm")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    output_csv = args.output_csv or os.path.join(args.output_dir, f"{args.prefix}_image_summary.csv")
    output_json = args.output_json or os.path.join(args.output_dir, f"{args.prefix}_summary.json")
    layer_csv = os.path.join(args.output_dir, f"{args.prefix}_layer_margin.csv")
    heatmap_png = os.path.join(args.output_dir, f"{args.prefix}_delta_margin_heatmap.png")

    data = load_jsonl(args.input_file)

    all_trace = []
    image_rows = []

    for item in data:
        image_id = item.get("image_id")
        caption = item.get("caption", "")
        trace = item.get("cbm_analysis", [])
        all_trace.extend(trace)

        stats = collect_stats(trace)
        image_rows.append({
            "image_id": image_id,
            "num_steps": stats["num_steps"],
            "pre_existence_score": stats["pre_existence_score"],
            "post_existence_score": stats["post_existence_score"],
            "delta_existence_score": stats["delta_existence_score"],
            "pre_safe_score": stats["pre_safe_score"],
            "post_safe_score": stats["post_safe_score"],
            "delta_safe_score": stats["delta_safe_score"],
            "pre_margin_safe_minus_existence": stats["pre_margin_safe_minus_existence"],
            "post_margin_safe_minus_existence": stats["post_margin_safe_minus_existence"],
            "delta_margin_safe_minus_existence": stats["delta_margin_safe_minus_existence"],
            "pre_existence_prob": stats["pre_existence_prob"],
            "post_existence_prob": stats["post_existence_prob"],
            "delta_existence_prob": stats["delta_existence_prob"],
            "router_edit_ratio": stats["router_edit_ratio"],
            "caption": caption,
        })

    overall = collect_stats(all_trace)
    edited_only = collect_stats(all_trace, filter_action=1)
    unedited_only = collect_stats(all_trace, filter_action=0)

    pre_layer_avg = average_layer_values(overall["_pre_layer_margin"])
    post_layer_avg = average_layer_values(overall["_post_layer_margin"])
    delta_layer_avg = average_layer_values(overall["_delta_layer_margin"])

    summary = {
        "num_images": len(data),
        "overall": strip_private_keys(overall),
        "router_action_1_edited_only": strip_private_keys(edited_only),
        "router_action_0_unedited_only": strip_private_keys(unedited_only),
        "layer_margin": {
            "pre_margin_safe_minus_existence": pre_layer_avg,
            "post_margin_safe_minus_existence": post_layer_avg,
            "delta_margin_safe_minus_existence": delta_layer_avg,
        },
        "interpretation_hint": {
            "delta_existence_score": "negative is desirable if HIRE reduces existence-risk concept",
            "delta_safe_score": "positive is desirable if HIRE moves representation toward safe concept",
            "delta_margin_safe_minus_existence": "positive means post-edit representation is more safe-dominant than pre-edit",
        },
    }

    print(json.dumps(summary, indent=2))

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if image_rows:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=image_rows[0].keys())
            writer.writeheader()
            writer.writerows(image_rows)

    save_layer_csv(layer_csv, pre_layer_avg, post_layer_avg, delta_layer_avg)
    save_margin_heatmap(
        heatmap_png,
        delta_layer_avg,
        "Layer-wise Δ Margin Heatmap: post(safe-existence) - pre(safe-existence)",
    )

    print(f"\nSaved summary json: {output_json}")
    print(f"Saved image csv: {output_csv}")
    print(f"Saved layer margin csv: {layer_csv}")
    print(f"Saved heatmap: {heatmap_png}")


if __name__ == "__main__":
    main()