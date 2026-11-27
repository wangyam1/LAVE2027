"""Single-GPU sequential blind runner for Qwen2.5-VL-7B-Instruct main experiments (max_steps=8, >=128 samples each).

Runs three frozen blind benchmarks on ONE GPU without reloading the model:
  1. Video-Holmes   (N=140 unique videos, 7x20)
  2. Video-MME      (N=144 unique videos, 48 Short + 48 Medium + 48 Long, all 12 tasks)
  3. LongVideoBench (N=150 questions / 146 long videos, all 17 categories)
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
from .refresh_mech_b import refresh_suite_mech_b

MODEL_PATH = Path("/data1/llj/development/model/Qwen/Qwen2.5-VL-7B-Instruct")
ROOT = Path(".")

BENCHMARKS = {
    "holmes140": dict(
        manifest="preflight_eval140.json",
        expected_pin="148f64d0e0745ff656bb28cda54654c950dee005bb5dc6df5ab67560501ccbbd",
        limit=140,
        adapter="holmes",
    ),
    "holmes315": dict(
        manifest="preflight_holmes315.json",
        expected_pin="cc7b42b1c87804770764587ec350d2146295dbd282fa4839ca82ad76628df087",
        limit=315,
        adapter="holmes",
    ),
    "mme144": dict(
        manifest="preflight_mme.json",
        expected_pin="5ce1fd95620eb747692001dadcddfeac3addbc0c35307d1b741d9c10b9075726",
        limit=144,
        adapter="mme",
    ),
    "mme306": dict(
        manifest="preflight_mme306.json",
        expected_pin="46b4df3266d252e85542a4d10fcb8f8c882b6f6bc0fa62a4cf1346e5076a2768",
        limit=306,
        adapter="mme",
    ),
    "lvb150": dict(
        manifest="preflight_lvb150.json",
        expected_pin="27b9d8338d198ec2bcd18d846ce7eaf3bae024d244a93692507c67b194d8ec08",
        limit=150,
        adapter="lvb",
    ),
    "lvb306": dict(
        manifest="preflight_lvb306.json",
        expected_pin="c03b726540f1e25573c60677d119a34edfa6e4e74df4797b26704df14e15d77a",
        limit=306,
        adapter="lvb",
    ),
}


def adapt_case(case, adapter):
    if adapter == "lvb":
        return dict(
            id=case["id"],
            video_id=case["video_uid"],
            question_category=case["question_category"],
            video_path=case["video_path"],
            duration=float(case["duration"]),
            question=case["question"],
            options=list(case["choices"]),
        )
    return case


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--suites", default="holmes140,mme144,lvb150")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--mamba-cache-mode", default="auto")
    ap.add_argument("--enable-thinking", action="store_true")
    ap.add_argument("--refresh-mech-b", action="store_true")
    args = ap.parse_args()

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    for s in suites:
        if s not in BENCHMARKS:
            raise ValueError(f"Unknown suite: {s}")

    # Verify all preflight manifests & pins upfront
    loaded = {}
    total_pending = 0
    for s in suites:
        spec = BENCHMARKS[s]
        pfile = ROOT / spec["manifest"]
        actual_pin = sha256_bytes(pfile.read_bytes())
        if spec["expected_pin"] is not None and actual_pin != spec["expected_pin"]:
            raise RuntimeError(f"Pin mismatch for {s}: {actual_pin} != {spec['expected_pin']}")
        manifest = json.loads(pfile.read_bytes())
        if manifest.get("labels") != 0:
            raise RuntimeError(f"Manifest {s} must be strictly blind (labels=0)")
        cases = manifest["cases"][: spec["limit"]]
        case_dir = ROOT / "action" / f"{s}_s{args.max_steps}"
        case_dir.mkdir(parents=True, exist_ok=True)
        done_rows = {}
        pending = []
        for ordinal, case in enumerate(cases):
            cpath = case_dir / f"case_{ordinal:03d}.json"
            if cpath.exists():
                done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
            else:
                pending.append((ordinal, case))
        total_pending += len(pending)
        loaded[s] = dict(spec=spec, pin=actual_pin, cases=cases, case_dir=case_dir, done_rows=done_rows, pending=pending)
        print(
            json.dumps(
                dict(
                    status="suite_queued",
                    suite=s,
                    model=args.model,
                    gpu=str(args.gpu),
                    max_steps=int(args.max_steps),
                    total=len(cases),
                    resumed=len(done_rows),
                    pending=len(pending),
                    pid=os.getpid(),
                ),
                sort_keys=True,
            ),
            flush=True,
        )

    engine = None
    runner = None
    if total_pending > 0 or args.refresh_mech_b:
        mc_arg = str(args.mamba_cache_mode).lower()
        if mc_arg == "auto":
            mamba_mode = "align" if "qwen3.5" in str(args.model).lower() else None
        elif mc_arg in ("none", ""):
            mamba_mode = None
        else:
            mamba_mode = args.mamba_cache_mode
        engine = MultiStepVLLMLaveEngine(
            args.model,
            max_model_len=24576,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=float(args.gpu_mem_util),
            max_num_seqs=64,
            enable_prefix_caching=True,
            mamba_cache_mode=mamba_mode,
            enforce_eager=False,
            mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 262144}},
        )
        cfg = AgentConfig(max_steps=int(args.max_steps), enable_thinking=bool(args.enable_thinking))
        runner = VideoSeekLaveRunner(engine, cfg)
        if args.refresh_mech_b:
            for s in suites:
                info = loaded[s]
                adapted_cases = [adapt_case(c, info["spec"]["adapter"]) for c in info["cases"]]
                refreshed = refresh_suite_mech_b(engine, cfg, adapted_cases, info["case_dir"], s)
                if refreshed > 0:
                    # Recompute done_rows hashes after atomic Option-B refresh
                    for ordinal, case in enumerate(info["cases"]):
                        cpath = info["case_dir"] / f"case_{ordinal:03d}.json"
                        if cpath.exists():
                            info["done_rows"][case["id"]] = sha256_bytes(cpath.read_bytes())

    deadline_epoch = float(os.environ.get("LAVE_DEADLINE_EPOCH", "0") or "0")
    stopped_early = False

    for s in suites:
        info = loaded[s]
        spec = info["spec"]
        case_dir = info["case_dir"]
        done_rows = info["done_rows"]
        pending = info["pending"]
        suite_stopped = False

        for ordinal, case in pending:
            if deadline_epoch > 0 and time() >= deadline_epoch:
                print(json.dumps(dict(status="deadline_reached", suite=s, done=len(done_rows), total=len(info["cases"]))), flush=True)
                stopped_early = True
                suite_stopped = True
                break
            t0 = perf_counter()
            adapted = adapt_case(case, spec["adapter"])
            rec = runner.run_case_paired(adapted, ordinal)
            rec["suite"] = s
            rec["model_path"] = args.model
            rec["max_steps"] = int(args.max_steps)
            if spec["adapter"] == "mme":
                rec["duration_bucket"] = case["duration_bucket"]
                rec["domain"] = case["domain"]
            elif spec["adapter"] == "lvb":
                rec["duration_group"] = int(case["duration_group"])
            cpath = case_dir / f"case_{ordinal:03d}.json"
            csha = write_json_canonical(cpath, rec)
            done_rows[case["id"]] = csha
            v_inv = sum(1 for st in rec["vanilla"]["steps"] if st["tool"] == "invalid")
            l_inv = sum(1 for st in rec["lave"]["steps"] if st["tool"] == "invalid")
            print(
                json.dumps(
                    dict(
                        status="case_committed",
                        suite=s,
                        done=len(done_rows),
                        total=len(info["cases"]),
                        ordinal=ordinal,
                        cid=case["id"],
                        cat=case["question_category"],
                        v_steps=rec["vanilla"]["num_steps"],
                        v_vcalls=rec["vanilla"]["visual_tool_calls"],
                        v_frames=rec["vanilla"]["total_frames"],
                        v_inv=v_inv,
                        l_steps=rec["lave"]["num_steps"],
                        l_vcalls=rec["lave"]["visual_tool_calls"],
                        l_frames=rec["lave"]["total_frames"],
                        l_inv=l_inv,
                        reroutes=rec["lave"]["reroute_count"],
                        rx_s=rec["lave"]["receiver_seconds"],
                        elapsed_s=round(perf_counter() - t0, 2),
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )

        if not suite_stopped:
            term = dict(
                status="complete",
                suite=s,
                model=args.model,
                max_steps=int(args.max_steps),
                pin=info["pin"],
                labels=0,
                gpu_idx=str(args.gpu),
                pid=os.getpid(),
                cases=len(done_rows),
                rows=done_rows,
            )
            tsha = write_json_canonical(case_dir / "terminal.json", term)
            print(json.dumps(dict(status="suite_complete", suite=s, terminal_sha256=tsha, cases=len(done_rows))), flush=True)
        if stopped_early:
            break


if __name__ == "__main__":
    main()
