"""Multi-GPU sharded blind runner for frozen LongVideoBench-150 using multi-step VideoSeek + LAVE (vLLM 0.19.0)."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig, VideoSeekLaveRunner
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import sha256_bytes, write_json_canonical

MODEL_PATH = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
ROOT = Path(".")
ACTION = ROOT / "action" / "lvb150"
FROZEN_LVB_PIN = "27b9d8338d198ec2bcd18d846ce7eaf3bae024d244a93692507c67b194d8ec08"


def run_shard(expected_pin, shard_id, num_shards, gpu_idx, gpu_mem_util=0.90, max_steps=6):
    pfile = ROOT / "preflight_lvb150.json"
    actual_pin = sha256_bytes(pfile.read_bytes())
    if actual_pin != expected_pin:
        raise RuntimeError(f"Preflight pin mismatch: {actual_pin} != {expected_pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("status") != "lvb150_vllm_preflight_frozen" or manifest.get("n") != 150:
        raise RuntimeError("Invalid frozen LongVideoBench-150 preflight manifest")

    shard_id = int(shard_id)
    num_shards = int(num_shards)
    ACTION.mkdir(parents=True, exist_ok=True)

    assigned = [
        (ordinal, case)
        for ordinal, case in enumerate(manifest["cases"])
        if ordinal % num_shards == shard_id
    ]
    done_rows = {}
    pending = []
    for ordinal, case in assigned:
        cpath = ACTION / f"case_{ordinal:03d}.json"
        if cpath.exists():
            done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
        else:
            pending.append((ordinal, case))

    print(
        json.dumps(
            dict(
                status="shard_boot",
                dataset="LongVideoBench-150",
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

    stopped_early = False
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
        deadline_epoch = float(os.environ.get("LAVE_DEADLINE_EPOCH", "0") or "0")
        for ordinal, case in pending:
            if deadline_epoch > 0 and time() >= deadline_epoch:
                print(
                    json.dumps(dict(status="deadline_reached", shard=shard_id, done=len(done_rows), assigned=len(assigned))),
                    flush=True,
                )
                stopped_early = True
                break
            t0 = perf_counter()
            adapted = dict(
                id=case["id"],
                video_id=case["video_uid"],
                question_category=case["question_category"],
                video_path=case["video_path"],
                duration=float(case["duration"]),
                question=case["question"],
                options=list(case["choices"]),
            )
            rec = runner.run_case_paired(adapted, ordinal)
            rec["duration_group"] = int(case["duration_group"])
            cpath = ACTION / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        dataset="LongVideoBench-150",
                        shard=shard_id,
                        done=len(done_rows),
                        assigned=len(assigned),
                        ordinal=ordinal,
                        cid=case["id"],
                        dg=int(case["duration_group"]),
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

    if not stopped_early:
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
        tsha = write_json_canonical(ACTION / f"shard_{shard_id}.terminal.json", term)
        print(
            json.dumps(dict(status="shard_complete", shard=shard_id, terminal_sha256=tsha, cases=len(done_rows))),
            flush=True,
        )


if __name__ == "__main__":
    run_shard(
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
        gpu_mem_util=float(sys.argv[5]) if len(sys.argv) > 5 else 0.90,
        max_steps=int(sys.argv[6]) if len(sys.argv) > 6 else 6,
    )
