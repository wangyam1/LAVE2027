"""Stage 3 unified matched-cohort ablation runner on Video-Holmes-315 (N=315, max_steps=8, single GPU).

Runs causal ablations on the EXACT SAME frozen blind manifest (`preflight_holmes315.json`, PIN cc7b42b1...):
  - Trajectory ablations (sharing Step-1 overview + tool_cache per case for zero-waste fairness):
      * clock_split     : Equal-budget 50/50 geometric split + Uniform-8 zero-pixel readout (clean & visonly)
      * no_calib        : LAVE w/o zero-embedding calibration (raw z_Y - z_N)
      * no_gate         : LAVE w/o single-video 3-control gate (always split)
      * fixed_k1        : LAVE with fixed K=1 peak (no entropy-guided multi-interval allocation)
      * lave_msweep     : Production LAVE with zero-extra-vision M in {4, 8, 16} readout sweep
  - Direct raw-pixel baseline (standalone, no captions):
      * videoseek_image : Interleaved raw-frame VideoSeek-Image (max_steps=8, zero-refusal)
  - Unshared standalone end-to-end latency audit (Table 6b):
      * unshared_latency: Balanced n=60 subset, zero shared overview, zero tool_cache, stage breakdown
"""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import perf_counter, time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

from .agent import AgentConfig, VideoSeekLaveRunner
from .baselines_holmes import ClockSplitRunner, VideoSeekImageRunner, run_standalone_unshared_latency
from .engine_ext import MultiStepVLLMLaveEngine
from .holmes_inputs import sha256_bytes, write_json_canonical

HOLMES315_PIN = "cc7b42b1c87804770764587ec350d2146295dbd282fa4839ca82ad76628df087"
TRAJ_ARMS = ("clock_split", "no_calib", "no_gate", "fixed_k1", "lave_msweep")
ALL_ARMS = TRAJ_ARMS + ("videoseek_image", "unshared_latency")


def select_balanced_ordinals(cases, target_n=60):
    """Deterministically select a category-balanced subset of ordinals from holmes315."""
    by_cat = defaultdict(list)
    for idx, c in enumerate(cases):
        by_cat[c["question_category"]].append(idx)
    cats = sorted(by_cat.keys())
    picked = []
    pos = 0
    while len(picked) < min(int(target_n), len(cases)):
        added = False
        for cat in cats:
            if pos < len(by_cat[cat]) and len(picked) < int(target_n):
                picked.append(by_cat[cat][pos])
                added = True
        if not added:
            break
        pos += 1
    return sorted(picked)


def make_ablation_config(arm_name, max_steps=8):
    """Build exact AgentConfig for each causal ablation arm."""
    if arm_name == "no_calib":
        return AgentConfig(max_steps=max_steps, use_zero_calibration=False)
    if arm_name == "no_gate":
        return AgentConfig(max_steps=max_steps, use_self_control_gate=False)
    if arm_name == "fixed_k1":
        return AgentConfig(max_steps=max_steps, fixed_max_peaks=1)
    if arm_name == "lave_msweep":
        return AgentConfig(max_steps=max_steps, mech_b_sweep_sizes=(4, 8, 16))
    return AgentConfig(max_steps=max_steps)


def run_ablation_bundle_for_case(engine, case, ordinal, req_traj_arms, *, max_steps=8):
    """Run all requested trajectory ablation arms for one case sharing Step-1 overview & tool_cache."""
    t0 = perf_counter()
    base_cfg = AgentConfig(max_steps=max_steps)
    base_runner = VideoSeekLaveRunner(engine, base_cfg)
    shared_ov = base_runner.run_shared_overview(case["video_path"], float(case["duration"]))
    tool_cache = {}

    arms_out = {}
    answers = {}
    for arm in req_traj_arms:
        if arm == "clock_split":
            runner = ClockSplitRunner(engine, base_cfg)
            res = runner.run_clock_split(case, shared_ov, tool_cache=tool_cache)
            arms_out[arm] = res
            answers["clock_split_text"] = res["text_answer"]
            answers["clock_split_step4"] = res["step4_text_answer"]
            answers["clock_split_step6"] = res["step6_text_answer"]
            answers["clock_split_uniform_qa"] = res["uniform_qa"]["answer"]
            answers["clock_split_uniform_visonly"] = res["uniform_qa"].get("answer_visual_only")
        else:
            cfg = make_ablation_config(arm, max_steps=max_steps)
            runner = VideoSeekLaveRunner(engine, cfg)
            res = runner._run_trajectory(
                case, shared_ov, use_lave_routing=True, seed_base=ordinal, tool_cache=tool_cache
            )
            arms_out[arm] = res
            answers[f"{arm}_text"] = res["text_answer"]
            answers[f"{arm}_step4"] = res["step4_text_answer"]
            answers[f"{arm}_step6"] = res["step6_text_answer"]
            answers[f"{arm}_clean_obs"] = res["latent_qa"]["answer"]
            answers[f"{arm}_visonly"] = res["latent_qa"].get("answer_visual_only")
            if arm == "lave_msweep" and "sweep_visonly" in res["latent_qa"]:
                for m_str, sw in res["latent_qa"]["sweep_visonly"].items():
                    answers[f"lave_visonly_m{m_str}"] = sw["answer"]

    return dict(
        cid=case["id"],
        ordinal=ordinal,
        video_id=case["video_id"],
        question_category=case["question_category"],
        duration=case["duration"],
        max_steps=int(max_steps),
        shared_overview=dict(
            frames=shared_ov["frames"],
            vision_calls=shared_ov["vision_calls"],
            seconds=shared_ov["seconds"],
        ),
        arms=arms_out,
        answers=answers,
        elapsed_seconds=round(perf_counter() - t0, 3),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data1/llj/development/model/Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--tag", default="qwen25vl_7b")
    ap.add_argument("--manifest", default="preflight_holmes315.json")
    ap.add_argument("--pin", default=HOLMES315_PIN)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--gpu-mem-util", type=float, default=0.90)
    ap.add_argument("--mamba-cache-mode", default="auto")
    ap.add_argument("--arms", default="videoseek_image,unshared_latency,clock_split,no_calib,no_gate,fixed_k1,lave_msweep")
    ap.add_argument("--latency-n", type=int, default=60)
    ap.add_argument("--limit", type=int, default=315)
    args = ap.parse_args()

    req_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    for a in req_arms:
        if a not in ALL_ARMS:
            raise ValueError(f"Unknown ablation arm: {a}")

    pfile = Path(args.manifest)
    actual_pin = sha256_bytes(pfile.read_bytes())
    if args.pin and actual_pin != args.pin:
        raise RuntimeError(f"Manifest PIN mismatch: {actual_pin} != {args.pin}")
    manifest = json.loads(pfile.read_bytes())
    if manifest.get("labels") != 0:
        raise RuntimeError("Ablation manifest must be strictly blind (labels=0)")
    cases = manifest["cases"][: int(args.limit)]

    root_dir = Path("action") / f"ablation_{args.tag}_s{args.max_steps}"
    root_dir.mkdir(parents=True, exist_ok=True)
    req_traj = [a for a in TRAJ_ARMS if a in req_arms]
    run_img = "videoseek_image" in req_arms
    run_lat = "unshared_latency" in req_arms

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
        limit_mm_per_prompt={"image": 64 if run_img else 32},
        enable_prefix_caching=True,
        mamba_cache_mode=mamba_mode,
        enforce_eager=False,
        mm_processor_kwargs={"size": {"shortest_edge": 65536, "longest_edge": 262144}},
    )

    deadline_epoch = float(os.environ.get("LAVE_DEADLINE_EPOCH", "0") or "0")

    def _run_img_stage():
        img_dir = root_dir / "videoseek_image"
        img_dir.mkdir(parents=True, exist_ok=True)
        img_runner = VideoSeekImageRunner(engine, AgentConfig(max_steps=int(args.max_steps)))
        done_rows = {}
        stopped = False
        for ordinal, case in enumerate(cases):
            cpath = img_dir / f"case_{ordinal:03d}.json"
            if cpath.exists():
                done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
                continue
            if deadline_epoch > 0 and time() >= deadline_epoch:
                stopped = True
                break
            res = img_runner.run_case_standalone(case)
            rec = dict(
                cid=case["id"],
                ordinal=ordinal,
                video_id=case["video_id"],
                question_category=case["question_category"],
                duration=case["duration"],
                max_steps=int(args.max_steps),
                model_path=args.model,
                videoseek_image=res,
                answers=dict(
                    videoseek_image=res["answer"],
                    videoseek_image_step4=res["step4_answer"],
                    videoseek_image_step6=res["step6_answer"],
                ),
            )
            done_rows[case["id"]] = write_json_canonical(cpath, rec)
            print(
                json.dumps(
                    dict(
                        status="img_case_committed",
                        tag=args.tag,
                        ordinal=ordinal,
                        done=len(done_rows),
                        total=len(cases),
                        cid=case["id"],
                        frames=res["total_frames"],
                        fed_frames=res["total_frames_fed_vlm"],
                        pruned=res["pruned_frames"],
                        elapsed_s=res["elapsed_seconds"],
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        if not stopped and len(done_rows) == len(cases):
            write_json_canonical(
                img_dir / "terminal.json",
                dict(
                    status="complete",
                    tag=args.tag,
                    model=args.model,
                    max_steps=int(args.max_steps),
                    pin=actual_pin,
                    labels=0,
                    n=len(done_rows),
                    pid=os.getpid(),
                    rows=done_rows,
                ),
            )

    def _run_lat_stage():
        lat_dir = root_dir / "unshared_latency"
        lat_dir.mkdir(parents=True, exist_ok=True)
        lat_ordinals = select_balanced_ordinals(cases, target_n=int(args.latency_n))
        cfg = AgentConfig(max_steps=int(args.max_steps))
        done_rows = {}
        stopped = False
        for ordinal in lat_ordinals:
            case = cases[ordinal]
            cpath = lat_dir / f"case_{ordinal:03d}.json"
            if cpath.exists():
                done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
                continue
            if deadline_epoch > 0 and time() >= deadline_epoch:
                stopped = True
                break
            res = run_standalone_unshared_latency(engine, cfg, case, ordinal)
            rec = dict(
                cid=case["id"],
                ordinal=ordinal,
                question_category=case["question_category"],
                duration=case["duration"],
                max_steps=int(args.max_steps),
                model_path=args.model,
                **res,
            )
            done_rows[case["id"]] = write_json_canonical(cpath, rec)
            print(
                json.dumps(
                    dict(
                        status="lat_case_committed",
                        tag=args.tag,
                        ordinal=ordinal,
                        done=len(done_rows),
                        total=len(lat_ordinals),
                        cid=case["id"],
                        van_s=res["vanilla_text_s"],
                        lav_rt_s=res["lave_route_text_s"],
                        lav_vis_s=res["lave_full_visonly_s"],
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        if not stopped and len(done_rows) == len(lat_ordinals):
            write_json_canonical(
                lat_dir / "terminal.json",
                dict(
                    status="complete",
                    tag=args.tag,
                    model=args.model,
                    max_steps=int(args.max_steps),
                    pin=actual_pin,
                    labels=0,
                    n=len(done_rows),
                    ordinals=lat_ordinals,
                    pid=os.getpid(),
                    rows=done_rows,
                ),
            )

    def _run_traj_stage():
        traj_dir = root_dir / "traj_bundle"
        traj_dir.mkdir(parents=True, exist_ok=True)
        done_rows = {}
        stopped = False
        for ordinal, case in enumerate(cases):
            cpath = traj_dir / f"case_{ordinal:03d}.json"
            if cpath.exists():
                existing = json.loads(cpath.read_bytes())
                if all(a in existing.get("arms", {}) for a in req_traj):
                    done_rows[case["id"]] = sha256_bytes(cpath.read_bytes())
                    continue
            if deadline_epoch > 0 and time() >= deadline_epoch:
                stopped = True
                break
            rec = run_ablation_bundle_for_case(engine, case, ordinal, req_traj, max_steps=int(args.max_steps))
            rec["model_path"] = args.model
            done_rows[case["id"]] = write_json_canonical(cpath, rec)
            print(
                json.dumps(
                    dict(
                        status="traj_case_committed",
                        tag=args.tag,
                        ordinal=ordinal,
                        done=len(done_rows),
                        total=len(cases),
                        cid=case["id"],
                        arms=req_traj,
                        elapsed_s=rec["elapsed_seconds"],
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
        if not stopped and len(done_rows) == len(cases):
            write_json_canonical(
                traj_dir / "terminal.json",
                dict(
                    status="complete",
                    tag=args.tag,
                    model=args.model,
                    max_steps=int(args.max_steps),
                    pin=actual_pin,
                    labels=0,
                    arms=req_traj,
                    n=len(done_rows),
                    pid=os.getpid(),
                    rows=done_rows,
                ),
            )

    # Execute stages in the order they first appear in --arms
    executed = set()
    for a in req_arms:
        if a == "videoseek_image" and "img" not in executed:
            executed.add("img")
            _run_img_stage()
        elif a == "unshared_latency" and "lat" not in executed:
            executed.add("lat")
            _run_lat_stage()
        elif a in TRAJ_ARMS and "traj" not in executed:
            executed.add("traj")
            _run_traj_stage()


if __name__ == "__main__":
    main()
