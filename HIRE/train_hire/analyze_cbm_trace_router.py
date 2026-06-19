import argparse
import json
import os
from collections import Counter, defaultdict


HALLU_CONCEPTS = ["existence", "attribute", "relation"]


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_score(trace_item, concept):
    return trace_item.get("concept_score", {}).get(concept, 0.0)


def get_max_hallu_score(trace_item):
    scores = {c: get_score(trace_item, c) for c in HALLU_CONCEPTS}
    best_c = max(scores, key=scores.get)
    return best_c, scores[best_c], scores


def analyze_item(item, safe_th, hallu_th):
    traces = item.get("cbm_analysis", [])

    risk_steps = []
    concept_counter = Counter()
    router_edit_count = 0
    cbm_risk_count = 0
    overlap_count = 0

    for t in traces:
        safe_score = get_score(t, "safe")
        best_hallu_concept, best_hallu_score, hallu_scores = get_max_hallu_score(t)

        router_action = t.get("router_action", None)
        router_edit = bool(router_action) if router_action is not None else False

        cbm_risk = (safe_score < safe_th) or (best_hallu_score > hallu_th)

        if router_edit:
            router_edit_count += 1
        if cbm_risk:
            cbm_risk_count += 1
            concept_counter[best_hallu_concept] += 1
        if router_edit and cbm_risk:
            overlap_count += 1

        if cbm_risk:
            risk_steps.append({
                "step": t.get("step"),
                "layer": t.get("layer"),
                "router_action": router_action,
                "pred_concept": t.get("pred_concept"),
                "safe_score": safe_score,
                "best_hallu_concept": best_hallu_concept,
                "best_hallu_score": best_hallu_score,
                "hallu_scores": hallu_scores,
                "concept_score": t.get("concept_score", {}),
                "concept_prob": t.get("concept_prob", {}),
                "prototype_similarity": t.get("prototype_similarity", {}),
            })

    n = max(len(traces), 1)

    return {
        "image_id": item.get("image_id"),
        "image_name": item.get("image_name"),
        "caption": item.get("caption"),
        "num_steps": len(traces),
        "num_risk_steps": cbm_risk_count,
        "risk_ratio": cbm_risk_count / n,
        "router_edit_count": router_edit_count,
        "router_edit_ratio": router_edit_count / n,
        "router_cbm_overlap_count": overlap_count,
        "router_cbm_overlap_ratio": overlap_count / max(router_edit_count, 1),
        "risk_concept_count": dict(concept_counter),
        "risk_steps": risk_steps,
    }


def main(args):
    rows = load_jsonl(args.input_file)

    summaries = []
    global_counter = Counter()
    total_steps = 0
    total_risk = 0
    total_router_edit = 0
    total_overlap = 0

    for item in rows:
        s = analyze_item(item, args.safe_threshold, args.hallu_threshold)
        summaries.append(s)

        total_steps += s["num_steps"]
        total_risk += s["num_risk_steps"]
        total_router_edit += s["router_edit_count"]
        total_overlap += s["router_cbm_overlap_count"]

        global_counter.update(s["risk_concept_count"])

    summaries = sorted(summaries, key=lambda x: x["risk_ratio"], reverse=True)

    output = {
        "config": {
            "input_file": args.input_file,
            "safe_threshold": args.safe_threshold,
            "hallu_threshold": args.hallu_threshold,
        },
        "global_summary": {
            "num_images": len(rows),
            "total_steps": total_steps,
            "total_risk_steps": total_risk,
            "risk_step_ratio": total_risk / max(total_steps, 1),
            "total_router_edit_steps": total_router_edit,
            "router_edit_ratio": total_router_edit / max(total_steps, 1),
            "router_cbm_overlap_steps": total_overlap,
            "router_cbm_overlap_over_router": total_overlap / max(total_router_edit, 1),
            "risk_concept_count": dict(global_counter),
        },
        "top_risky_captions": summaries[:args.top_k],
        "all_summaries": summaries,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "cbm_trace_analysis.json")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\nGlobal summary")
    for k, v in output["global_summary"].items():
        print(f"{k}: {v}")

    print("\nTop risky captions")
    for s in summaries[:10]:
        print("=" * 80)
        print("image:", s["image_id"], s["image_name"])
        print("risk_ratio:", round(s["risk_ratio"], 4))
        print("risk_concept_count:", s["risk_concept_count"])
        print("caption:", s["caption"][:300])
        print("first risk steps:")
        for r in s["risk_steps"][:5]:
            print(
                f"  step={r['step']} "
                f"router={r['router_action']} "
                f"pred={r['pred_concept']} "
                f"safe={r['safe_score']:.3f} "
                f"{r['best_hallu_concept']}={r['best_hallu_score']:.3f}"
            )

    print("\nSaved:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="output/analysis/cbm_trace")

    parser.add_argument("--safe-threshold", type=float, default=0.5)
    parser.add_argument("--hallu-threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=50)

    args = parser.parse_args()
    main(args)