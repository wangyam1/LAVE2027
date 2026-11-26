#!/usr/bin/env bash
set -euo pipefail
umask 077

cd "$(dirname "$0")/.."
python=/data1/llj/anaconda3/envs/qwen35_vlm_tf513/bin/python
annotation=/data1/llj/development/lljvideo/CLAUDEvidoe/CG-Bench/cgbench.json
pid=$(cat supervisor.pid)
export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1

while status=$(ps -o stat= -p "$pid" 2>/dev/null) && [[ -n "$status" && "$status" != Z* ]]; do
    sleep 15
done
if ! "$python" -B -c 'import json; from pathlib import Path; p=Path("action/terminal.json"); raise SystemExit(not (p.is_file() and json.loads(p.read_bytes()).get("status")=="complete"))'; then
    printf '%s\n' "Inference ended before completion. The recorded prefix is preserved."
    exit 0
fi

"$python" -B -u -m confirm96_20261126.confirm_audit release
release=$(sha256sum release.json | cut -d ' ' -f 1)
"$python" -B -u -m confirm96_20261126.confirm_score "$release" "$annotation" scoring/metadata_projection_scanner.py > score.pending
mv score.pending score.json
sha256sum score.json
