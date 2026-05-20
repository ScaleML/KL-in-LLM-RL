#!/bin/bash
set -euo pipefail

source /root/.python/rl-kl-verl/bin/activate

NUM_SHARDS="${HOST_NUM:-1}"
SHARD_RANK="${INDEX:-0}"
RESULTS_DIR="${RESULTS_DIR:-~/rl-kl/verl/data/results}"
GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE:-16}"
MERGE_TIMEOUT="${MERGE_TIMEOUT:-1800}"

echo "[eval] shard ${SHARD_RANK}/${NUM_SHARDS} on ${HOSTNAME:-unknown}"
echo "[eval] results dir: ${RESULTS_DIR}"
echo "[eval] generate batch size: ${GENERATE_BATCH_SIZE}"
echo "[eval] merge timeout: ${MERGE_TIMEOUT}"

python3 eval/eval.py \
  --model <model_path> \
  --data aime24 \
  --data aime25 \
  --data olympiadbench \
  --data amc23 \
  --data minerva_math \
  --output-dir "${RESULTS_DIR}" \
  --output-name <output_name> \
  --n 16 \
  --temperature 1.0 \
  --max-response-length 8192 \
  --max-model-len 10240 \
  --tp 4 \
  --dp 2 \
  --generate-batch-size "${GENERATE_BATCH_SIZE}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-rank "${SHARD_RANK}" \
  --merge-shards \
  --merge-timeout "${MERGE_TIMEOUT}" \
  --disable-custom-all-reduce \
  --enforce-eager