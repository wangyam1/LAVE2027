"""Multi-GPU sharded blind runner for Video-MME multi-step VideoSeek vs VideoSeek+LAVE (vLLM 0.19.0)."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig, VideoSeekLaveRunner
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import sha256_bytes, write_json_canonical
from .mme_inputs import MME_PARQUET, MME_VIDEOS, build_balanced_mme_manifest

MODEL_PATH = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
ROOT = Path(".")
ACTION = ROOT / "action"


def preflight(per_duration=72):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run preflight on CPU with CUDA_VISIBLE_DEVICES=''")
    manifest = build_balanced_mme_manifest(MME_PARQUET, MME_VIDEOS, per_duration=int(per_duration))
    pfile = ROOT / "preflight_mme.json"
    sha = write_json_canonical(pfile, manifest)
    print(
        json.dumps(
            dict(
                status="preflight_written",
                path=str(pfile),
                pin=sha,
                n=manifest["n"],
                unique_videos=manifest["unique_videos"],
                duration_counts=manifest["duration_counts"],
                task_counts=manifest["task_counts"],
                domain_counts=manifest["domain_counts"],
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


def run_shard(expected_pin, shard_id, num_shards, gpu_idx, gpu_mem_util=0.90, max_steps=6):
    pfile = ROOT / "preflight_mme.json"
    actual_pin = sha256_bytes(pfile.read_bytes())
    if actual_pin != expected_pin:
        raise RuntimeError(f"Preflight pin mismatch: {actual_pin} != {expected_pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("labels") != 0:
        raise RuntimeError("Preflight manifest must be strictly blind (labels=0)")

    shard_id = int(shard_id)
    num_shards = int(num_shards)
    case_dir = ACTION / "mme"
    case_dir.mkdir(parents=True, exist_ok=True)

    assigned = [
        (ordinal, case)
        for ordinal, case in enumerate(manifest["cases"])
        if ordinal % num_shards == shard_id
    ]
    done_rows = {}
    pending = []
    for ordinal, case in assigned:
        cpath = case_dir / f"case_{ordinal:03d}.json"
        if cpath.exists():
            done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
        else:
            pending.append((ordinal, case))

    print(
        json.dumps(
            dict(
                status="shard_boot",
                shard=shard_id,
                num_shards=num_shards,
                gpu_idx=str(gpu_idx),
                assigned=len(assigned),
                resumed=len(done_rows),
                pending=len(pending),
                max_steps=int(max_steps),
                gpu_mem_util=float(gpu_mem_util),
                pid=os.getpid(),
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
            rec["duration_bucket"] = case["duration_bucket"]
            rec["domain"] = case["domain"]
            cpath = case_dir / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        shard=shard_id,
                        done=len(done_rows),
                        assigned=len(assigned),
                        ordinal=ordinal,
                        cid=case["id"],
                        dur=case["duration_bucket"],
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
        shard=shard_id,
        num_shards=num_shards,
        pin=actual_pin,
        labels=0,
        gpu_idx=str(gpu_idx),
        pid=os.getpid(),
        cases=len(done_rows),
        rows=done_rows,
    )
    tsha = write_json_canonical(case_dir / f"shard_{shard_id}.terminal.json", term)
    print(
        json.dumps(dict(status="shard_complete", shard=shard_id, terminal_sha256=tsha, cases=len(done_rows))),
        flush=True,
    )


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "preflight":
        preflight(int(sys.argv[2]) if len(sys.argv) > 2 else 72)
    elif cmd == "run":
        run_shard(
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            sys.argv[5],
            gpu_mem_util=float(sys.argv[6]) if len(sys.argv) > 6 else 0.90,
            max_steps=int(sys.argv[7]) if len(sys.argv) > 7 else 6,
        )
    else:
        raise SystemExit(f"Unknown command: {cmd}")
