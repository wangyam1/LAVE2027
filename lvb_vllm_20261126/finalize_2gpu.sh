#!/usr/bin/env bash
set -euo pipefail
umask 077

cd "$(dirname "$0")/.."
ENV_DST=/data1/llj/lljmas/envs/lave_vllm_20261126
PY="$ENV_DST/bin/python"
export LD_LIBRARY_PATH="$ENV_DST/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1

p0=$(cat shard0.pid)
p1=$(cat shard1.pid)

wait_pid() {
  local pid=$1
  while status=$(ps -o stat= -p "$pid" 2>/dev/null) && [[ -n "$status" && "$status" != Z* ]]; do
    sleep 15
  done
}

wait_pid "$p0"
wait_pid "$p1"

if ! "$PY" -B -c 'import json; from pathlib import Path; ok=all(Path(f"action/shard_{i}.terminal.json").is_file() and json.loads(Path(f"action/shard_{i}.terminal.json").read_bytes()).get("status")=="complete" for i in (0,1)); raise SystemExit(not ok)'; then
  printf '%s\n' "One or both shards ended before completion."
  exit 0
fi

"$PY" -B -u -m lvb_vllm_20261126.lvb_score 2 > score_stdout.log
sha256sum score.json
