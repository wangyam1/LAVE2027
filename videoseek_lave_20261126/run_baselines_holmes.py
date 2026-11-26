"""3-GPU sharded blind runner for Clock-Split, VideoSeek-Image, and standalone latency audit on Video-Holmes eval140."""

import json
import os
from pathlib import Path
import sys
from time import perf_counter, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig
from .baselines_holmes import ClockSplitRunner, VideoSeekImageRunner, run_standalone_unshared_latency
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import sha256_bytes, write_json_canonical

MODEL_PATH = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
ROOT = Path(".")
ACTION = ROOT / "action" / "baselines_eval140"


def run_shard(expected_pin, shard_id, num_shards, gpu_idx, gpu_mem_util=0.90, max_steps=6):
    pfile = ROOT / "preflight_eval140.json"
    actual_pin = sha256_bytes(pfile.read_bytes())
    if actual_pin != expected_pin:
        raise RuntimeError(f"Preflight pin mismatch: {actual_pin} != {expected_pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("labels") != 0:
        raise RuntimeError("Preflight manifest must be strictly blind (labels=0)")

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
                suite="holmes_eval140_baselines",
                shard=shard_id,
                num_shards=num_shards,
                gpu_idx=str(gpu_idx),
                assigned=len(assigned),
                resumed=len(done_rows),
                pending=len(pending),
                max_steps=int(max_steps),
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
            max_model_len=24576,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=float(gpu_mem_util),
            max_num_seqs=64,
            limit_mm_per_prompt={"image": 64},
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            enforce_eager=False,
            mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 262144}},
        )
        cfg = AgentConfig(max_steps=int(max_steps))
        clock_runner = ClockSplitRunner(engine, cfg)
        img_runner = VideoSeekImageRunner(engine, cfg, max_context_frames=64)
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
            shared_ov = clock_runner.run_shared_overview(case["video_path"], float(case["duration"]))
            clock_res = clock_runner.run_clock_split(case, shared_ov, tool_cache={})
            img_res = img_runner.run_case_standalone(case)
            latency_audit = None
            if ordinal % 5 == 0:
                latency_audit = run_standalone_unshared_latency(engine, cfg, case, ordinal)

            rec = dict(
                cid=case["id"],
                ordinal=ordinal,
                video_id=case["video_id"],
                question_category=case["question_category"],
                duration=case["duration"],
                clock_split=clock_res,
                videoseek_image=img_res,
                latency_audit=latency_audit,
                answers=dict(
                    clock_split_text=clock_res["text_answer"],
                    clock_split_step4=clock_res["step4_text_answer"],
                    clock_split_uniform_qa=clock_res["uniform_qa"]["answer"],
                    videoseek_image=img_res["answer"],
                    videoseek_image_step4=img_res["step4_answer"],
                ),
                elapsed_seconds=round(perf_counter() - t0, 3),
            )
            cpath = ACTION / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        suite="holmes_eval140_baselines",
                        shard=shard_id,
                        done=len(done_rows),
                        assigned=len(assigned),
                        ordinal=ordinal,
                        cid=case["id"],
                        cat=case["question_category"],
                        clk_frames=clock_res["total_frames"],
                        img_frames=img_res["total_frames"],
                        img_fed=img_res["total_frames_fed_vlm"],
                        img_s=img_res["elapsed_seconds"],
                        has_lat=latency_audit is not None,
                        elapsed_s=round(perf_counter() - t0, 2),
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )

    if not stopped_early:
        term = dict(
            status="complete",
            suite="holmes_eval140_baselines",
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
        print(json.dumps(dict(status="shard_complete", shard=shard_id, terminal_sha256=tsha, cases=len(done_rows))), flush=True)


if __name__ == "__main__":
    run_shard(
        sys.argv[1],
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
        gpu_mem_util=float(sys.argv[5]) if len(sys.argv) > 5 else 0.90,
        max_steps=int(sys.argv[6]) if len(sys.argv) > 6 else 6,
    )
