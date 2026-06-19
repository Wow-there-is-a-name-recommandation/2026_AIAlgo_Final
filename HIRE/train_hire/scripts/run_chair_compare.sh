#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

DATA_SIZE=100
BATCH_SIZE=16
CBM_BATCH_SIZE=1

MODEL_PATH="checkpoints/llava-v1.5-7b"
IMAGE_FOLDER="data/coco2014/val2014"
ANNO_FILE="data/coco2014/annotations/instances_val2014.json"
ANNO_DIR="data/coco2014/annotations"

EDITOR_PATH="output/checkpoint/hire_editor/hire_editor_final.pth"
ROUTER_PATH="output/RL_router/router_llava_epoch100_new_sgd_banlance.pth"
CBM_PATH="output/checkpoint/amber_layerwise_cbm_existence_generative/cbm_best.pt"

OUT_DIR="output/chair_compare_100"
mkdir -p "$OUT_DIR"

RUNTIME_CSV="$OUT_DIR/runtime_seconds.csv"
SUMMARY_CSV="$OUT_DIR/chair_summary.csv"

echo "model,seconds" > "$RUNTIME_CSV"

run_and_time () {
  name="$1"
  shift

  start=$(date +%s)
  echo "=== Generate $name START ==="
  "$@"
  end=$(date +%s)

  elapsed=$((end-start))
  echo "$name,$elapsed" >> "$RUNTIME_CSV"
  echo "=== Generate $name END: ${elapsed}s ==="
}

eval_chair () {
  name="$1"
  cap_file="$2"
  metric_file="$3"

  echo "=== Evaluate $name CHAIR ==="

  python - "$name" "$cap_file" "$metric_file" <<'PY'
import sys, json
from train_hire.CHAIR import CHAIR

name, cap_file, metric_file = sys.argv[1], sys.argv[2], sys.argv[3]

chair = CHAIR("data/coco2014/annotations")
result = chair.compute_chair(cap_file, "image_id", "caption")

print(name + ":", result["overall_metrics"])

with open(metric_file, "w") as f:
    json.dump(result, f, indent=2)
PY
}

# append 방지: 기존 파일 비우기
: > "$OUT_DIR/llava.jsonl"
: > "$OUT_DIR/hire.jsonl"
: > "$OUT_DIR/llava_cbm.jsonl"
: > "$OUT_DIR/hire_cbm.jsonl"

# 1. LLaVA
run_and_time "LLaVA" \
python train_hire/generate_captions_baseline.py \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --anno-folder "$ANNO_FILE" \
  --output-file "$OUT_DIR/llava.jsonl" \
  --data-size "$DATA_SIZE" \
  --num-chunks 1 \
  --batch-size "$BATCH_SIZE" \
  --chunk-idx 0

eval_chair "LLaVA" "$OUT_DIR/llava.jsonl" "$OUT_DIR/llava_metrics.json"

# 2. HIRE
run_and_time "HIRE" \
python train_hire/generate_captions_router.py \
  --editor-path "$EDITOR_PATH" \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --anno-folder "$ANNO_FILE" \
  --output-file "$OUT_DIR/hire.jsonl" \
  --data-size "$DATA_SIZE" \
  --num-chunks 1 \
  --edit-strength 1.0 \
  --batch-size "$BATCH_SIZE" \
  --chunk-idx 0

eval_chair "HIRE" "$OUT_DIR/hire.jsonl" "$OUT_DIR/hire_metrics.json"

# 3. LLaVA + Layer-wise CBM
run_and_time "LLaVA_CBM" \
python train_hire/generate_captions_with_llava_layerwise_exis_cbm_analysis.py \
  --model-path "$MODEL_PATH" \
  --cbm-path "$CBM_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --anno-folder "$ANNO_FILE" \
  --output-file "$OUT_DIR/llava_cbm.jsonl" \
  --data-size "$DATA_SIZE" \
  --num-chunks 1 \
  --chunk-idx 0 \
  --batch-size "$CBM_BATCH_SIZE" \
  --max-new-tokens 512

eval_chair "LLaVA_CBM" "$OUT_DIR/llava_cbm.jsonl" "$OUT_DIR/llava_cbm_metrics.json"

# 4. HIRE + Layer-wise CBM
run_and_time "HIRE_CBM" \
python train_hire/generate_captions_with_layerwise_exis_cbm_analysis.py \
  --editor-path "$EDITOR_PATH" \
  --cbm-path "$CBM_PATH" \
  --model-path "$MODEL_PATH" \
  --image-folder "$IMAGE_FOLDER" \
  --anno-folder "$ANNO_FILE" \
  --output-file "$OUT_DIR/hire_cbm.jsonl" \
  --data-size "$DATA_SIZE" \
  --num-chunks 1 \
  --chunk-idx 0 \
  --batch-size "$CBM_BATCH_SIZE" \
  --edit-strength 1.0 \
  --max-new-tokens 512

eval_chair "HIRE_CBM" "$OUT_DIR/hire_cbm.jsonl" "$OUT_DIR/hire_cbm_metrics.json"

# 5. Summary 저장
python - <<'PY'
import json, csv

metric_files = {
    "LLaVA": "output/chair_compare_100/llava_metrics.json",
    "HIRE": "output/chair_compare_100/hire_metrics.json",
    "LLaVA_CBM": "output/chair_compare_100/llava_cbm_metrics.json",
    "HIRE_CBM": "output/chair_compare_100/hire_cbm_metrics.json",
}

runtime = {}
with open("output/chair_compare_100/runtime_seconds.csv") as f:
    next(f)
    for line in f:
        name, sec = line.strip().split(",")
        runtime[name] = int(sec)

rows = []
for name, path in metric_files.items():
    with open(path) as f:
        data = json.load(f)
    m = data["overall_metrics"]
    rows.append({
        "Model": name,
        "CHAIRs": m.get("CHAIRs"),
        "CHAIRi": m.get("CHAIRi"),
        "Recall": m.get("Recall"),
        "Len": m.get("Len"),
        "Runtime_sec": runtime.get(name),
    })

with open("output/chair_compare_100/chair_summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Model", "CHAIRs", "CHAIRi", "Recall", "Len", "Runtime_sec"])
    writer.writeheader()
    writer.writerows(rows)

print("\nModel\tCHAIRs\tCHAIRi\tRecall\tLen\tRuntime_sec")
for r in rows:
    print(f"{r['Model']}\t{r['CHAIRs']}\t{r['CHAIRi']}\t{r['Recall']}\t{r['Len']}\t{r['Runtime_sec']}")
PY

echo "Done."
echo "Summary saved to: $SUMMARY_CSV"
echo "Runtime saved to: $RUNTIME_CSV"