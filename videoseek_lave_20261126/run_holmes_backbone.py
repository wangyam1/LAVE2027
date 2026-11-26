"""Multi-GPU sharded blind runner for multi-backbone / Thinking-mode generalization on Video-Holmes eval140.

Supports:
- Qwen3.5-4B (scale generalization)
- Qwen3.5-9B with enable_thinking=True (Thinking-model generalization)
- Qwen2.5-VL-7B-Instruct (architecture-family generalization + exact LARVA backbone match)
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig, VideoSeekLaveRunner
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import sha256_bytes, write_json_canonical

ROOT = Path(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--num-shards", type=int, default=3)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--thinking", type=int, default=0)
    ap.add_argument("--no-mamba", type=int, default=0)
    args = ap.parse_args()

    pfile = ROOT / "preflight_eval140.json"
    actual_pin = sha256_bytes(pfile.read_bytes())
    if actual_pin != args.pin:
        raise RuntimeError(f"Preflight pin mismatch: {actual_pin} != {args.pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("labels") != 0:
        raise RuntimeError("Preflight manifest must be strictly blind (labels=0)")

    case_dir = ROOT / "action" / args.tag
    case_dir.mkdir(parents=True, exist_ok=True)

    assigned = [
        (ordinal, case)
        for ordinal, case in enumerate(manifest["cases"])
        if ordinal % args.num_shards == args.shard
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
                tag=args.tag,
                model=args.model,
                thinking=bool(args.thinking),
                no_mamba=bool(args.no_mamba),
                shard=args.shard,
                num_shards=args.num_shards,
                gpu_idx=str(args.gpu),
                assigned=len(assigned),
                resumed=len(done_rows),
                pending=len(pending),
                pid=os.getpid(),
            ),
            sort_keys=True,
        ),
        flush=True,
    )

    stopped_early = False
    if pending:
        mm_kw = {"size": {"shortest_edge": 65536, "longest_edge": 262144}}
        engine = MultiStepVLLMLaveEngine(
            args.model,
            max_model_len=20000,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=float(args.gpu_mem_util),
            max_num_seqs=64,
            enable_prefix_caching=True,
            mamba_cache_mode=None if args.no_mamba else "align",
            enforce_eager=False,
            mm_processor_kwargs=mm_kw,
        )
        cfg = AgentConfig(
            max_steps=int(args.max_steps),
            enable_thinking=bool(args.thinking),
            thinking_max_tokens=1024,
        )
        runner = VideoSeekLaveRunner(engine, cfg)
        deadline_epoch = float(os.environ.get("LAVE_DEADLINE_EPOCH", "0") or "0")

        for ordinal, case in pending:
            if deadline_epoch > 0 and time() >= deadline_epoch:
                print(
                    json.dumps(dict(status="deadline_reached", tag=args.tag, shard=args.shard, done=len(done_rows), assigned=len(assigned))),
                    flush=True,
                )
                stopped_early = True
                break
            t0 = perf_counter()
            rec = runner.run_case_paired(case, ordinal)
            rec["backbone_tag"] = args.tag
            rec["model_path"] = args.model
            rec["enable_thinking"] = bool(args.thinking)
            cpath = case_dir / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        tag=args.tag,
                        shard=args.shard,
                        done=len(done_rows),
                        assigned=len(assigned),
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

    if not stopped_early:
        term = dict(
            status="complete",
            tag=args.tag,
            model=args.model,
            thinking=bool(args.thinking),
            shard=args.shard,
            num_shards=args.num_shards,
            pin=actual_pin,
            labels=0,
            gpu_idx=str(args.gpu),
            pid=os.getpid(),
            cases=len(done_rows),
            rows=done_rows,
        )
        tsha = write_json_canonical(case_dir / f"shard_{args.shard}.terminal.json", term)
        print(json.dumps(dict(status="shard_complete", tag=args.tag, shard=args.shard, terminal_sha256=tsha, cases=len(done_rows))), flush=True)


if __name__ == "__main__":
    main()
