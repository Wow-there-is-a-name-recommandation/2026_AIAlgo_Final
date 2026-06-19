#!/bin/bash
set -e

# =========================================================
# Slack notification
# =========================================================
SLACK_WEBHOOK_URL=""

notify_slack () {
    MSG="$1"
    if [ -n "$SLACK_WEBHOOK_URL" ]; then
        curl -s -X POST \
            -H 'Content-type: application/json' \
            --data "{\"text\":\"${MSG}\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null || true
    fi
}

on_error () {
    notify_slack "❌ AMBER sharded extraction FAILED

Host: $(hostname)
Time: $(date)
Output: $HS_PATH

Check terminal logs."
}
trap on_error ERR

# =========================================================
# Config
# =========================================================
export CUDA_VISIBLE_DEVICES=0

MODEL_PATH="checkpoints/llava-v1.5-7b"
HS_PATH="output/hidden_states_amber_proto_shard"

FULL_JSON="data/amber_proto/amber_proto_full_concept.json"
GEN_JSON="data/amber_proto/amber_proto_generative_only.json"

BATCH_SIZE=2
STEP=500

get_len () {
python - <<PY
import json
print(len(json.load(open("$1"))))
PY
}

run_sharded_extract () {
    DATA_PATH="$1"
    SPLIT_NAME="$2"
    TOTAL=$(get_len "$DATA_PATH")

    notify_slack "🚀 START ${SPLIT_NAME}

Host: $(hostname)
Total: ${TOTAL}
Step: ${STEP}
Batch size: ${BATCH_SIZE}
Output: ${HS_PATH}
Time: $(date)"

    for START in $(seq 0 $STEP $((TOTAL - 1)))
    do
        END=$((START + STEP))
        if [ "$END" -gt "$TOTAL" ]; then
            END=$TOTAL
        fi

        notify_slack "🟡 Running ${SPLIT_NAME} shard ${START}-${END}"

        python train_hire/extract_hidden_states_amber.py \
            --model-path "$MODEL_PATH" \
            --data-path "$DATA_PATH" \
            --hs-path "$HS_PATH" \
            --split-name "$SPLIT_NAME" \
            --batch-size "$BATCH_SIZE" \
            --start-index "$START" \
            --end-index "$END" \
            --num-chunks 1 \
            --chunk-idx 0 || {
                notify_slack "❌ FAILED ${SPLIT_NAME} shard ${START}-${END}

Host: $(hostname)
Time: $(date)"
                exit 1
            }

        notify_slack "✅ Done ${SPLIT_NAME} shard ${START}-${END}"
    done

    notify_slack "🎉 FINISHED ${SPLIT_NAME}

Host: $(hostname)
Time: $(date)"
}

notify_slack "🟡 AMBER sharded extraction script STARTED

Host: $(hostname)
GPU: ${CUDA_VISIBLE_DEVICES}
Time: $(date)"

run_sharded_extract "$FULL_JSON" "full"
run_sharded_extract "$GEN_JSON" "generative_only"

notify_slack "🎉 ALL AMBER extraction FINISHED

Host: $(hostname)
Output: ${HS_PATH}
Time: $(date)"

echo "Done. Check outputs in: $HS_PATH"