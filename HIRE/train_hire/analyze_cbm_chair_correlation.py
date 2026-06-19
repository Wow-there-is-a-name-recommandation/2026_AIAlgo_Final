import argparse
import json
import os
from collections import defaultdict

import numpy as np


HALLU_CONCEPTS = ["existence", "attribute", "relation"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_chair_metrics(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sent = data.get("sentences", None)
    if sent is None:
        raise ValueError("CHAIR metrics json must contain result['sentences'].")

    chair_map = {}
    for row in sent:
        image_id = row.get("image_id", row.get("image_id".upper(), None))
        if image_id is None:
            image_id = row.get("image_id")

        metrics = row.get("metrics", {})
        chair_map[int(image_id)] = {
            "CHAIRs": metrics.get("CHAIRs", None),
            "CHAIRi": metrics.get("CHAIRi", None),
            "Recall": metrics.get("Recall", None),
            "Len": metrics.get("Len", None),
        }

    return chair_map


def score(trace, concept):
    return trace.get("concept_score", {}).get(concept, 0.0)


def summarize_cbm(item, safe_th, hallu_th):
    traces = item.get("cbm_analysis", [])
    n = max(len(traces), 1)

    risk_count = 0
    concept_count = defaultdict(int)

    safe_scores = []
    hallu_scores = []
    existence_scores = []
    attribute_scores = []
    relation_scores = []

    for t in traces:
        safe = score(t, "safe")
        existence = score(t, "existence")
        attribute = score(t, "attribute")
        relation = score(t, "relation")

        hallu_dict = {
            "existence": existence,
            "attribute": attribute,
            "relation": relation,
        }

        best_concept = max(hallu_dict, key=hallu_dict.get)
        best_hallu = hallu_dict[best_concept]

        is_risk = (safe < safe_th) or (best_hallu > hallu_th)

        if is_risk:
            risk_count += 1
            concept_count[best_concept] += 1

        safe_scores.append(safe)
        hallu_scores.append(best_hallu)
        existence_scores.append(existence)
        attribute_scores.append(attribute)
        relation_scores.append(relation)

    return {
        "num_steps": len(traces),
        "risk_count": risk_count,
        "risk_ratio": risk_count / n,
        "risk_concept_count": dict(concept_count),
        "safe_mean": float(np.mean(safe_scores)) if safe_scores else 0.0,
        "max_hallu_mean": float(np.mean(hallu_scores)) if hallu_scores else 0.0,
        "existence_mean": float(np.mean(existence_scores)) if existence_scores else 0.0,
        "attribute_mean": float(np.mean(attribute_scores)) if attribute_scores else 0.0,
        "relation_mean": float(np.mean(relation_scores)) if relation_scores else 0.0,
    }


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if len(x) < 2:
        return None
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None

    return float(np.corrcoef(x, y)[0, 1])


def group_stats(rows, key, chair_key="CHAIRi", threshold=0.0):
    high = [r for r in rows if r[chair_key] is not None and r[chair_key] > threshold]
    low = [r for r in rows if r[chair_key] is not None and r[chair_key] <= threshold]

    def avg(xs, k):
        if not xs:
            return None
        return float(np.mean([x[k] for x in xs]))

    return {
        "chair_key": chair_key,
        "threshold": threshold,
        "num_high_chair": len(high),
        "num_low_chair": len(low),
        f"high_{key}_mean": avg(high, key),
        f"low_{key}_mean": avg(low, key),
    }


def main(args):
    cbm_rows = load_jsonl(args.cbm_jsonl)
    chair_map = load_chair_metrics(args.chair_metrics)

    merged = []

    for item in cbm_rows:
        image_id = int(item["image_id"])
        if image_id not in chair_map:
            continue

        cbm_summary = summarize_cbm(
            item,
            safe_th=args.safe_threshold,
            hallu_th=args.hallu_threshold,
        )

        chair = chair_map[image_id]

        row = {
            "image_id": image_id,
            "image_name": item.get("image_name"),
            "caption": item.get("caption"),
            **cbm_summary,
            **chair,
        }

        merged.append(row)

    if not merged:
        raise RuntimeError("No matched image_id between CBM jsonl and CHAIR metrics.")

    correlation_keys = [
        "risk_ratio",
        "safe_mean",
        "max_hallu_mean",
        "existence_mean",
        "attribute_mean",
        "relation_mean",
    ]

    correlations = {}

    for ck in ["CHAIRi", "CHAIRs", "Recall"]:
        y = [r[ck] for r in merged if r.get(ck) is not None]
        correlations[ck] = {}

        for xk in correlation_keys:
            xs = [r[xk] for r in merged if r.get(ck) is not None]
            ys = [r[ck] for r in merged if r.get(ck) is not None]
            correlations[ck][xk] = pearson(xs, ys)

    top_by_chairi = sorted(
        merged,
        key=lambda r: -1 if r["CHAIRi"] is None else r["CHAIRi"],
        reverse=True,
    )[:args.top_k]

    top_by_risk = sorted(
        merged,
        key=lambda r: r["risk_ratio"],
        reverse=True,
    )[:args.top_k]

    group_analysis = {
        "chairi_gt_0": {
            k: group_stats(merged, k, chair_key="CHAIRi", threshold=0.0)
            for k in correlation_keys
        },
        "chairs_gt_0": {
            k: group_stats(merged, k, chair_key="CHAIRs", threshold=0.0)
            for k in correlation_keys
        },
    }

    output = {
        "config": {
            "cbm_jsonl": args.cbm_jsonl,
            "chair_metrics": args.chair_metrics,
            "safe_threshold": args.safe_threshold,
            "hallu_threshold": args.hallu_threshold,
        },
        "summary": {
            "num_matched": len(merged),
            "avg_CHAIRi": float(np.mean([r["CHAIRi"] for r in merged if r["CHAIRi"] is not None])),
            "avg_CHAIRs": float(np.mean([r["CHAIRs"] for r in merged if r["CHAIRs"] is not None])),
            "avg_risk_ratio": float(np.mean([r["risk_ratio"] for r in merged])),
            "avg_safe_mean": float(np.mean([r["safe_mean"] for r in merged])),
            "avg_existence_mean": float(np.mean([r["existence_mean"] for r in merged])),
            "avg_attribute_mean": float(np.mean([r["attribute_mean"] for r in merged])),
            "avg_relation_mean": float(np.mean([r["relation_mean"] for r in merged])),
        },
        "correlations": correlations,
        "group_analysis": group_analysis,
        "top_by_CHAIRi": top_by_chairi,
        "top_by_CBM_risk": top_by_risk,
        "all": merged,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "cbm_chair_correlation.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nSummary")
    for k, v in output["summary"].items():
        print(f"{k}: {v}")

    print("\nCorrelations")
    for chair_k, row in correlations.items():
        print(f"[{chair_k}]")
        for k, v in row.items():
            print(f"  {k}: {v}")

    print("\nGroup analysis: CHAIRi > 0 vs CHAIRi = 0")
    for k, v in group_analysis["chairi_gt_0"].items():
        print(k, v)

    print("\nTop by CHAIRi")
    for r in top_by_chairi[:10]:
        print("=" * 80)
        print("image:", r["image_id"])
        print("CHAIRi:", r["CHAIRi"], "CHAIRs:", r["CHAIRs"], "risk:", r["risk_ratio"])
        print("means:",
              "safe", round(r["safe_mean"], 3),
              "exist", round(r["existence_mean"], 3),
              "attr", round(r["attribute_mean"], 3),
              "rel", round(r["relation_mean"], 3))
        print("caption:", r["caption"][:250])

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cbm-jsonl", type=str, required=True)
    parser.add_argument("--chair-metrics", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="output/analysis/cbm_chair")

    parser.add_argument("--safe-threshold", type=float, default=0.5)
    parser.add_argument("--hallu-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()
    main(args)