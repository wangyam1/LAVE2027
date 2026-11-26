#!/usr/bin/env bash
set -euo pipefail

NUM_SHARDS="${1:-3}"
PY="/data1/llj/lljmas/envs/lave_vllm_20261126/bin/python"
export LD_LIBRARY_PATH="/data1/llj/lljmas/envs/lave_vllm_20261126/lib:${LD_LIBRARY_PATH:-}"

for sid in $(seq 0 $((NUM_SHARDS - 1))); do
  if [ -f "shard${sid}.pid" ]; then
    PID=$(cat "shard${sid}.pid")
    while kill -0 "$PID" 2>/dev/null; do
      sleep 10
    done
  fi
done
sleep 5

echo "=== GPU STATUS AFTER ALL LVB SHARDS EXIT ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

CUDA_VISIBLE_DEVICES="" "$PY" -B -u -m videoseek_lave_20261126.score_lvb "$NUM_SHARDS" > score_lvb.log 2>&1
sha256sum score_lvb150_multistep.json
