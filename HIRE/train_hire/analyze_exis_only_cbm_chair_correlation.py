import argparse
import json
import os
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_chair_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chair_map = {}
    for row in data["sentences"]:
        image_id = int(row["image_id"])
        m = row.get("metrics", {})
        chair_map[image_id] = {
            "CHAIRs": m.get("CHAIRs"),
            "CHAIRi": m.get("CHAIRi"),
            "Recall": m.get("Recall"),
            "Len": m.get("Len"),
        }
    return chair_map


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def group_stats(rows, score_key, chair_key, threshold=0.0):
    high = [r for r in rows if r[chair_key] is not None and r[chair_key] > threshold]
    low = [r for r in rows if r[chair_key] is not None and r[chair_key] <= threshold]

    def avg(xs, k):
        return None if not xs else float(np.mean([x[k] for x in xs]))

    return {
        "chair_key": chair_key,
        "score_key": score_key,
        "num_high_chair": len(high),
        "num_low_chair": len(low),
        "high_mean": avg(high, score_key),
        "low_mean": avg(low, score_key),
    }


def main(args):
    cbm_rows = load_jsonl(args.cbm_jsonl)
    chair_map = load_chair_metrics(args.chair_metrics)

    merged = []

    for item in cbm_rows:
        image_id = int(item["image_id"])
        if image_id not in chair_map:
            continue

        s = item.get("existence_summary", {})

        row = {
            "image_id": image_id,
            "image_name": item.get("image_name"),
            "caption": item.get("caption"),
            "num_steps": s.get("num_steps", 0),
            "avg_existence_score": s.get("avg_existence_score", 0.0),
            "max_existence_score": s.get("max_existence_score", 0.0),
            "top5_existence_score": s.get("top5_existence_score", 0.0),
            "existence_risk_ratio": s.get("existence_risk_ratio", 0.0),
            "avg_existence_prob": s.get("avg_existence_prob", 0.0),
            "max_existence_prob": s.get("max_existence_prob", 0.0),
            "avg_existence_proto_sim": s.get("avg_existence_proto_sim", 0.0),
            "max_existence_proto_sim": s.get("max_existence_proto_sim", 0.0),
            **chair_map[image_id],
        }
        merged.append(row)

    if not merged:
        raise RuntimeError("No matched image_id between CBM jsonl and CHAIR metrics.")

    score_keys = [
        "avg_existence_score",
        "max_existence_score",
        "top5_existence_score",
        "existence_risk_ratio",
        "avg_existence_prob",
        "max_existence_prob",
        "avg_existence_proto_sim",
        "max_existence_proto_sim",
    ]

    correlations = {}
    for chair_key in ["CHAIRi", "CHAIRs", "Recall"]:
        correlations[chair_key] = {}
        valid = [r for r in merged if r.get(chair_key) is not None]
        for score_key in score_keys:
            correlations[chair_key][score_key] = pearson(
                [r[score_key] for r in valid],
                [r[chair_key] for r in valid],
            )

    group_analysis = {
        "CHAIRi_gt_0": {
            k: group_stats(merged, k, "CHAIRi", 0.0) for k in score_keys
        },
        "CHAIRs_gt_0": {
            k: group_stats(merged, k, "CHAIRs", 0.0) for k in score_keys
        },
    }

    top_by_chairi = sorted(
        merged,
        key=lambda r: -1 if r["CHAIRi"] is None else r["CHAIRi"],
        reverse=True,
    )[:args.top_k]

    top_by_risk = sorted(
        merged,
        key=lambda r: r["existence_risk_ratio"],
        reverse=True,
    )[:args.top_k]

    output = {
        "config": vars(args),
        "summary": {
            "num_matched": len(merged),
            "avg_CHAIRi": float(np.mean([r["CHAIRi"] for r in merged if r["CHAIRi"] is not None])),
            "avg_CHAIRs": float(np.mean([r["CHAIRs"] for r in merged if r["CHAIRs"] is not None])),
            "avg_existence_score": float(np.mean([r["avg_existence_score"] for r in merged])),
            "avg_existence_risk_ratio": float(np.mean([r["existence_risk_ratio"] for r in merged])),
        },
        "correlations": correlations,
        "group_analysis": group_analysis,
        "top_by_CHAIRi": top_by_chairi,
        "top_by_existence_risk": top_by_risk,
        "all": merged,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "exis_only_cbm_chair_correlation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nSummary")
    for k, v in output["summary"].items():
        print(f"{k}: {v}")

    print("\nCorrelations")
    for chair_key, row in correlations.items():
        print(f"[{chair_key}]")
        for k, v in row.items():
            print(f"  {k}: {v}")

    print("\nGroup analysis: CHAIRi > 0")
    for k, v in group_analysis["CHAIRi_gt_0"].items():
        print(k, v)

    print("\nTop by CHAIRi")
    for r in top_by_chairi[:10]:
        print("=" * 80)
        print("image:", r["image_id"])
        print("CHAIRi:", r["CHAIRi"], "CHAIRs:", r["CHAIRs"])
        print("avg:", round(r["avg_existence_score"], 3),
              "max:", round(r["max_existence_score"], 3),
              "top5:", round(r["top5_existence_score"], 3),
              "risk:", round(r["existence_risk_ratio"], 3))
        print("caption:", r["caption"][:250])

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbm-jsonl", type=str, required=True)
    parser.add_argument("--chair-metrics", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="output/analysis/exis_only_cbm_chair")
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()
    main(args)