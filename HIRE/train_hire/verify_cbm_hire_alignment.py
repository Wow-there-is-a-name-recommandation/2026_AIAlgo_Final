import argparse
import csv
import json
import os
import math
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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def pearson(x, y):
    if len(x) < 2:
        return 0.0
    mx, my = mean(x), mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx == 0 or deny == 0:
        return 0.0
    return num / (denx * deny)


def extract_chair_per_image(chair_json):
    candidates = []

    if isinstance(chair_json, dict):
        for key in ["sentences", "per_image", "image_metrics", "results", "details"]:
            if key in chair_json and isinstance(chair_json[key], list):
                candidates = chair_json[key]
                break

    elif isinstance(chair_json, list):
        candidates = chair_json

    mapping = {}

    for item in candidates:
        if not isinstance(item, dict):
            continue

        image_id = item.get("image_id", item.get("imageId", item.get("id")))
        if image_id is None:
            continue

        metrics = item.get("metrics", {})

        chairi = (
            item.get("CHAIRi")
            if "CHAIRi" in item else
            metrics.get("CHAIRi", metrics.get("chairi", metrics.get("CHAIR_i", None)))
        )

        chairs = (
            item.get("CHAIRs")
            if "CHAIRs" in item else
            metrics.get("CHAIRs", metrics.get("chairs", metrics.get("CHAIR_s", None)))
        )

        recall = (
            item.get("Recall")
            if "Recall" in item else
            metrics.get("Recall", metrics.get("recall", None))
        )

        mapping[int(image_id)] = {
            "CHAIRi": chairi,
            "CHAIRs": chairs,
            "Recall": recall,
        }

    return mapping


def summarize_dual_image(item):
    trace = item.get("cbm_analysis", [])

    delta_margins = []
    pre_margins = []
    post_margins = []
    delta_exis = []
    delta_safe = []

    pre_exis_layer = []
    post_exis_layer = []
    pre_safe_layer = []
    post_safe_layer = []

    for x in trace:
        px = x.get("pre_existence_score", 0.0)
        qx = x.get("post_existence_score", 0.0)
        ps = x.get("pre_safe_score", 0.0)
        qs = x.get("post_safe_score", 0.0)

        pre_m = ps - px
        post_m = qs - qx

        pre_margins.append(pre_m)
        post_margins.append(post_m)
        delta_margins.append(post_m - pre_m)
        delta_exis.append(x.get("delta_existence_score", qx - px))
        delta_safe.append(x.get("delta_safe_score", qs - ps))

        if x.get("pre_existence_layer_proto_sim"):
            pre_exis_layer.append(x["pre_existence_layer_proto_sim"])
        if x.get("post_existence_layer_proto_sim"):
            post_exis_layer.append(x["post_existence_layer_proto_sim"])
        if x.get("pre_safe_layer_proto_sim"):
            pre_safe_layer.append(x["pre_safe_layer_proto_sim"])
        if x.get("post_safe_layer_proto_sim"):
            post_safe_layer.append(x["post_safe_layer_proto_sim"])

    return {
        "image_id": int(item["image_id"]),
        "caption": item.get("caption", ""),
        "num_steps": len(trace),
        "pre_margin": mean(pre_margins),
        "post_margin": mean(post_margins),
        "delta_margin": mean(delta_margins),
        "delta_existence_score": mean(delta_exis),
        "delta_safe_score": mean(delta_safe),
        "pre_existence_layer": avg_layer(pre_exis_layer),
        "post_existence_layer": avg_layer(post_exis_layer),
        "pre_safe_layer": avg_layer(pre_safe_layer),
        "post_safe_layer": avg_layer(post_safe_layer),
    }


def summarize_single_cbm_image(item):
    trace = item.get("cbm_analysis", [])

    exis_scores = []
    safe_scores = []
    exis_probs = []
    margins = []
    exis_layer = []
    safe_layer = []

    for x in trace:
        ex = x.get("existence_score", 0.0)
        sf = x.get("safe_score", 0.0)
        exis_scores.append(ex)
        safe_scores.append(sf)
        margins.append(sf - ex)
        exis_probs.append(x.get("existence_prob", 0.0))

        if x.get("existence_layer_proto_sim"):
            exis_layer.append(x["existence_layer_proto_sim"])
        if x.get("safe_layer_proto_sim"):
            safe_layer.append(x["safe_layer_proto_sim"])

    return {
        "image_id": int(item["image_id"]),
        "caption": item.get("caption", ""),
        "num_steps": len(trace),
        "existence_score": mean(exis_scores),
        "safe_score": mean(safe_scores),
        "existence_prob": mean(exis_probs),
        "margin_safe_minus_existence": mean(margins),
        "existence_layer": avg_layer(exis_layer),
        "safe_layer": avg_layer(safe_layer),
    }


def avg_layer(layer_rows):
    if not layer_rows:
        return []
    n = len(layer_rows[0])
    out = []
    for i in range(n):
        vals = [r[i] for r in layer_rows if len(r) > i]
        out.append(mean(vals))
    return out


def avg_layer_across_images(rows, key):
    vals = [r[key] for r in rows if r.get(key)]
    return avg_layer(vals)


def save_scatter(path, x, y, xlabel, ylabel, title):
    plt.figure(figsize=(5, 4))
    plt.scatter(x, y, s=18, alpha=0.75)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_hist(path, values, xlabel, title):
    plt.figure(figsize=(5, 4))
    plt.hist(values, bins=30)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_layer_line(path, series_dict, ylabel, title):
    plt.figure(figsize=(8, 4))
    for name, vals in series_dict.items():
        if vals:
            plt.plot(list(range(len(vals))), vals, marker="o", label=name)
    plt.xlabel("Layer")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_heatmap(path, values, title):
    if not values:
        return
    plt.figure(figsize=(12, 2.2))
    plt.imshow([values], aspect="auto")
    plt.yticks([0], ["value"])
    plt.xticks(range(len(values)), range(len(values)), rotation=90)
    plt.xlabel("Layer")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    # 검증 1: dual CBM pre/post 결과 + CHAIR per-image metric
    parser.add_argument("--dual-file", required=True)
    parser.add_argument("--chair-metrics", required=True)

    # 검증 2: LLaVA+CBM, HIRE+CBM single-CBM 결과 비교
    parser.add_argument("--llava-cbm-file", required=True)
    parser.add_argument("--hire-cbm-file", required=True)

    parser.add_argument("--output-dir", default="output/cbm_hire_verification")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------
    # 검증 1: image별 delta_margin vs CHAIRi
    # -------------------------
    dual_rows_raw = load_jsonl(args.dual_file)
    dual_rows = [summarize_dual_image(x) for x in dual_rows_raw]

    chair = extract_chair_per_image(load_json(args.chair_metrics))

    corr_rows = []
    xs_delta_margin = []
    ys_chairi = []

    for r in dual_rows:
        image_id = r["image_id"]
        if image_id not in chair:
            continue

        chairi = chair[image_id].get("CHAIRi")
        chairs = chair[image_id].get("CHAIRs")

        if chairi is None:
            continue

        corr_rows.append({
            "image_id": image_id,
            "delta_margin": r["delta_margin"],
            "delta_existence_score": r["delta_existence_score"],
            "delta_safe_score": r["delta_safe_score"],
            "CHAIRi": chairi,
            "CHAIRs": chairs,
            "caption": r["caption"],
        })
        xs_delta_margin.append(r["delta_margin"])
        ys_chairi.append(chairi)

    corr_delta_margin_chairi = pearson(xs_delta_margin, ys_chairi)

    with open(os.path.join(args.output_dir, "verify1_delta_margin_vs_chair.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "delta_margin", "delta_existence_score", "delta_safe_score", "CHAIRi", "CHAIRs", "caption"],
        )
        writer.writeheader()
        writer.writerows(corr_rows)

    save_scatter(
        os.path.join(args.output_dir, "verify1_delta_margin_vs_chairi.png"),
        xs_delta_margin,
        ys_chairi,
        "Image-level Δ margin: post(safe-existence) - pre(safe-existence)",
        "CHAIRi",
        f"Δ margin vs CHAIRi (r={corr_delta_margin_chairi:.3f})",
    )

    # -------------------------
    # 검증 2: LLaVA vs HIRE CBM score
    # -------------------------
    llava_rows = [summarize_single_cbm_image(x) for x in load_jsonl(args.llava_cbm_file)]
    hire_rows = [summarize_single_cbm_image(x) for x in load_jsonl(args.hire_cbm_file)]

    llava_map = {r["image_id"]: r for r in llava_rows}
    hire_map = {r["image_id"]: r for r in hire_rows}

    compare_rows = []
    for image_id in sorted(set(llava_map) & set(hire_map)):
        l = llava_map[image_id]
        h = hire_map[image_id]

        compare_rows.append({
            "image_id": image_id,
            "llava_existence_score": l["existence_score"],
            "hire_existence_score": h["existence_score"],
            "delta_hire_minus_llava_existence": h["existence_score"] - l["existence_score"],

            "llava_safe_score": l["safe_score"],
            "hire_safe_score": h["safe_score"],
            "delta_hire_minus_llava_safe": h["safe_score"] - l["safe_score"],

            "llava_margin": l["margin_safe_minus_existence"],
            "hire_margin": h["margin_safe_minus_existence"],
            "delta_hire_minus_llava_margin": h["margin_safe_minus_existence"] - l["margin_safe_minus_existence"],
        })

    with open(os.path.join(args.output_dir, "verify2_llava_vs_hire_cbm.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=compare_rows[0].keys())
        writer.writeheader()
        writer.writerows(compare_rows)

    # -------------------------
    # 검증 3: layer-wise prototype 분포 / histogram / layer curve
    # -------------------------
    dual_pre_ex_layer = avg_layer_across_images(dual_rows, "pre_existence_layer")
    dual_post_ex_layer = avg_layer_across_images(dual_rows, "post_existence_layer")
    dual_pre_safe_layer = avg_layer_across_images(dual_rows, "pre_safe_layer")
    dual_post_safe_layer = avg_layer_across_images(dual_rows, "post_safe_layer")

    dual_delta_ex_layer = [b - a for a, b in zip(dual_pre_ex_layer, dual_post_ex_layer)]
    dual_delta_safe_layer = [b - a for a, b in zip(dual_pre_safe_layer, dual_post_safe_layer)]
    dual_delta_margin_layer = [
        (post_s - post_e) - (pre_s - pre_e)
        for pre_e, post_e, pre_s, post_s in zip(
            dual_pre_ex_layer, dual_post_ex_layer, dual_pre_safe_layer, dual_post_safe_layer
        )
    ]

    save_layer_line(
        os.path.join(args.output_dir, "verify3_pre_post_existence_layer_curve.png"),
        {"pre_existence": dual_pre_ex_layer, "post_existence": dual_post_ex_layer},
        "Prototype similarity",
        "HIRE pre/post existence prototype similarity",
    )

    save_layer_line(
        os.path.join(args.output_dir, "verify3_pre_post_safe_layer_curve.png"),
        {"pre_safe": dual_pre_safe_layer, "post_safe": dual_post_safe_layer},
        "Prototype similarity",
        "HIRE pre/post safe prototype similarity",
    )

    save_heatmap(
        os.path.join(args.output_dir, "verify3_delta_margin_layer_heatmap.png"),
        dual_delta_margin_layer,
        "Layer-wise Δ margin: post(safe-existence) - pre(safe-existence)",
    )

    save_hist(
        os.path.join(args.output_dir, "verify3_image_delta_margin_hist.png"),
        [r["delta_margin"] for r in dual_rows],
        "Image-level Δ margin",
        "Distribution of image-level Δ margin",
    )

    summary = {
        "verify1_delta_margin_vs_CHAIRi": {
            "num_matched_images": len(corr_rows),
            "pearson_r": corr_delta_margin_chairi,
            "interpretation": "If r is positive, larger safe-margin shift aligns with larger CHAIRi. If r is negative, smaller/negative margin shift aligns with lower CHAIRi, suggesting CBM axis may be opposite to CHAIR improvement.",
        },
        "verify2_llava_vs_hire_cbm": {
            "num_matched_images": len(compare_rows),
            "mean_llava_existence_score": mean([r["llava_existence_score"] for r in compare_rows]),
            "mean_hire_existence_score": mean([r["hire_existence_score"] for r in compare_rows]),
            "mean_delta_hire_minus_llava_existence": mean([r["delta_hire_minus_llava_existence"] for r in compare_rows]),
            "mean_llava_safe_score": mean([r["llava_safe_score"] for r in compare_rows]),
            "mean_hire_safe_score": mean([r["hire_safe_score"] for r in compare_rows]),
            "mean_delta_hire_minus_llava_safe": mean([r["delta_hire_minus_llava_safe"] for r in compare_rows]),
            "mean_llava_margin": mean([r["llava_margin"] for r in compare_rows]),
            "mean_hire_margin": mean([r["hire_margin"] for r in compare_rows]),
            "mean_delta_hire_minus_llava_margin": mean([r["delta_hire_minus_llava_margin"] for r in compare_rows]),
        },
        "verify3_layerwise": {
            "delta_existence_layer": dual_delta_ex_layer,
            "delta_safe_layer": dual_delta_safe_layer,
            "delta_margin_layer": dual_delta_margin_layer,
        },
    }

    with open(os.path.join(args.output_dir, "verification_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()