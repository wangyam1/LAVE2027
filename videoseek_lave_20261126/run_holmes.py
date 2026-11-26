"""Single-GPU blind runner for multi-step VideoSeek vs VideoSeek+LAVE on Video-Holmes (vLLM 0.19.0)."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig, VideoSeekLaveRunner
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import (
    HOLMES_JSON,
    HOLMES_VIDEOS,
    build_balanced_holmes_manifests,
    sha256_bytes,
    write_json_canonical,
)

MODEL_PATH = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
ROOT = Path(".")
ACTION = ROOT / "action"


def preflight(split="pilot14"):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run preflight on CPU with CUDA_VISIBLE_DEVICES=''")
    pilot, eval_m = build_balanced_holmes_manifests(HOLMES_JSON, HOLMES_VIDEOS)
    manifest = pilot if split == "pilot14" else eval_m
    pfile = ROOT / f"preflight_{split}.json"
    sha = write_json_canonical(pfile, manifest)
    print(
        json.dumps(
            dict(
                status="preflight_written",
                split=split,
                path=str(pfile),
                pin=sha,
                n=manifest["n"],
                unique_videos=manifest["unique_videos"],
                category_counts=manifest["category_counts"],
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def run_split(split, expected_pin, gpu_idx="1", gpu_mem_util=0.90, max_steps=6):
    pfile = ROOT / f"preflight_{split}.json"
    actual_pin = sha256_bytes(pfile.read_bytes())
    if actual_pin != expected_pin:
        raise RuntimeError(f"Preflight pin mismatch: {actual_pin} != {expected_pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("labels") != 0:
        raise RuntimeError("Preflight manifest must be strictly blind (labels=0)")

    case_dir = ACTION / split
    case_dir.mkdir(parents=True, exist_ok=True)
    done_rows = {}
    pending = []
    for ordinal, case in enumerate(manifest["cases"]):
        cpath = case_dir / f"case_{ordinal:03d}.json"
        if cpath.exists():
            done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
        else:
            pending.append((ordinal, case))

    print(
        json.dumps(
            dict(
                status="run_start",
                split=split,
                gpu_idx=gpu_idx,
                total=len(manifest["cases"]),
                resumed=len(done_rows),
                pending=len(pending),
                max_steps=max_steps,
                gpu_mem_util=gpu_mem_util,
            ),
            sort_keys=True,
        ),
        flush=True,
    )

    if pending:
        engine = MultiStepVLLMLaveEngine(
            MODEL_PATH,
            max_model_len=20000,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=float(gpu_mem_util),
            max_num_seqs=64,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            enforce_eager=False,
            mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 262144}},
        )
        runner = VideoSeekLaveRunner(engine, AgentConfig(max_steps=int(max_steps)))
        for ordinal, case in pending:
            t0 = perf_counter()
            rec = runner.run_case_paired(case, ordinal)
            cpath = case_dir / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        split=split,
                        done=len(done_rows),
                        total=len(manifest["cases"]),
                        ordinal=ordinal,
                        cid=case["id"],
                        cat=case["question_category"],
                        v_steps=rec["vanilla"]["num_steps"],
                        v_frames=rec["vanilla"]["total_frames"],
                        l_steps=rec["lave"]["num_steps"],
                        l_frames=rec["lave"]["total_frames"],
                        reroutes=rec["lave"]["reroute_count"],
                        rx_s=rec["lave"]["receiver_seconds"],
                        elapsed_s=round(perf_counter() - t0, 2),
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )

    term = dict(
        status="complete",
        split=split,
        pin=actual_pin,
        labels=0,
        gpu_idx=str(gpu_idx),
        pid=os.getpid(),
        cases=len(done_rows),
        rows=done_rows,
    )
    tsha = write_json_canonical(case_dir / "terminal.json", term)
    print(json.dumps(dict(status="split_complete", split=split, terminal_sha256=tsha, cases=len(done_rows))), flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "preflight":
        preflight(sys.argv[2] if len(sys.argv) > 2 else "pilot14")
    elif cmd == "run":
        run_split(
            sys.argv[2],
            sys.argv[3],
            gpu_idx=sys.argv[4] if len(sys.argv) > 4 else "1",
            gpu_mem_util=float(sys.argv[5]) if len(sys.argv) > 5 else 0.90,
            max_steps=int(sys.argv[6]) if len(sys.argv) > 6 else 6,
        )
    else:
        raise SystemExit(f"Unknown command: {cmd}")
