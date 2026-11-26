"""Benchmark vLLM LAVE numerical parity & latency against audited HF dev receipts.

Runs strictly on the 4 exposed development probe cases (ordinals 16, 6, 20, 38) using the
exact sealed 32 input JPEGs and reports already persisted in ``entropy_routing_20261125``,
requiring zero label reads, zero decord dependency, and zero changes to existing receipts.
"""

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from entropy_routing_20261125 import policy
from missing_fact_routing_20261124.target import target_from_question

from .engine import VLLMLaveEngine


DEV_ENTROPY = Path("/data1/llj/lljmas/qwen35_cgbench_entropy_routing_20261125_01")
DEFAULT_MODEL = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
DEFAULT_ORDINALS = (16, 6, 20, 38)
NATIVE_MIN_PIXELS = 65536
NATIVE_MAX_PIXELS = 262144


def _margin(pair):
    return float(pair[0]) - float(pair[1])


def _compare_scan(hf_scan, vllm_scan):
    hf_margins = [_margin(p) for p in hf_scan["real"]]
    vl_margins = [_margin(p) for p in vllm_scan["real"]]
    diffs = [abs(a - b) for a, b in zip(hf_margins, vl_margins)]
    zero_diff = abs(_margin(hf_scan["zero"]) - _margin(vllm_scan["zero"]))
    hf_lifts = [a - _margin(hf_scan["zero"]) for a in hf_margins]
    vl_lifts = [a - _margin(vllm_scan["zero"]) for a in vl_margins]
    lift_diffs = [abs(a - b) for a, b in zip(hf_lifts, vl_lifts)]
    return dict(
        max_margin_diff=round(max(diffs), 5),
        mean_margin_diff=round(sum(diffs) / len(diffs), 5),
        zero_margin_diff=round(zero_diff, 5),
        max_lift_diff=round(max(lift_diffs), 5),
        mean_lift_diff=round(sum(lift_diffs) / len(lift_diffs), 5),
    )


def _resolve_model_path(explicit):
    if explicit is not None:
        return Path(explicit)
    if DEFAULT_MODEL.is_dir():
        return DEFAULT_MODEL
    # Fallback: read model path from native ITT block_02 environment receipt if present
    env_file = Path("/data1/llj/lljmas/qwen35_native_itt_fresh96_20261116_04/blocks/block_02/protocol_manifest.json")
    if env_file.is_file():
        meta = json.loads(env_file.read_bytes())
        for key in ("model_path", "model"):
            if key in meta and Path(meta[key]).is_dir():
                return Path(meta[key])
    raise FileNotFoundError("Pass --model-path pointing to local Qwen3.5-9B directory")


def run_benchmark(
    ordinals,
    *,
    model_path=None,
    batch_size=33,
    gpu_mem_util=0.50,
    mamba_cache_mode="align",
    enforce_eager=False,
    min_pixels=NATIVE_MIN_PIXELS,
    max_pixels=NATIVE_MAX_PIXELS,
):
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Expose one idle A40 GPU via CUDA_VISIBLE_DEVICES")
    from PIL import Image

    resolved_model = _resolve_model_path(model_path)
    mm_kwargs = dict(size={"shortest_edge": min_pixels, "longest_edge": max_pixels})
    t_boot = perf_counter()
    engine = VLLMLaveEngine(
        resolved_model,
        max_model_len=20000,
        gpu_memory_utilization=gpu_mem_util,
        enable_prefix_caching=True,
        mamba_cache_mode=mamba_cache_mode,
        enforce_eager=enforce_eager,
        mm_processor_kwargs=mm_kwargs,
    )
    boot_s = round(perf_counter() - t_boot, 2)

    dev_preflight = json.loads((DEV_ENTROPY / "preflight.json").read_bytes())
    rows_out = []

    for ordinal in ordinals:
        case_dir = DEV_ENTROPY / "action" / f"case_{ordinal:02d}"
        hf_case = json.loads((case_dir / "case.json").read_bytes())
        cid = hf_case["cid"]
        hf_audit = hf_case["route_audit"]
        if hf_audit is None:
            raise ValueError(f"Case {ordinal} ({cid}) had no HF route_audit")

        src_shared_dir = case_dir / "source_shared"
        src_shared_rec = json.loads((src_shared_dir / "outputs" / "source_shared.json").read_bytes())
        src_query_rec = json.loads((case_dir / "source_query" / "outputs" / "source_query.json").read_bytes())
        hf_reports = [src_shared_rec["text"], src_query_rec["text"]]

        # Load the exact 32 sealed JPEGs fed to HF source_shared for bit-exact visual parity
        pil_images = []
        for img_meta in src_shared_rec["images"]:
            with Image.open(src_shared_dir / img_meta["path"]) as im:
                pil_images.append(im.convert("RGB"))

        # 1. Live vLLM source_shared with single-pass visual embedding capture
        src_out = engine.generate_source_shared(src_shared_rec["messages"], pil_images, max_tokens=2048)

        # 2. Exact HF-matched Receiver parity check on the captured 32 slots (both batched and batch=1)
        text_act = hf_case["text_action"]
        target = hf_case["target"]
        foil_q = dev_preflight["foils"][cid]["question"]
        wrong_target = target_from_question(foil_q)
        hf_grid, = {tuple(g) for g in src_shared_rec["image_grid_thw"]}
        vl_grid, = set(src_out["grids"])

        eligible = list(range(31))
        if hf_audit.get("decision") and "eligible" in hf_audit["decision"]:
            eligible = list(hf_audit["decision"]["eligible"])

        vllm_scan = engine.probe_states(
            target, text_act["query"], hf_reports, vl_grid, src_out["slots"], batch_size=batch_size,
        )
        vllm_scan_b1 = (
            vllm_scan if batch_size == 1 else engine.probe_states(
                target, text_act["query"], hf_reports, vl_grid, src_out["slots"], batch_size=1,
            )
        )
        cand = policy.nominate(vllm_scan["real"], vllm_scan["zero"], eligible)
        vllm_ctrl = None
        vllm_dec = None
        if cand is not None:
            vllm_ctrl = engine.probe_controls(
                ordinal, target, wrong_target, text_act["query"], hf_reports, vl_grid, src_out["slots"], cand.slot,
            )
            vllm_dec = policy.approve(
                cand, vllm_scan["real"], vllm_scan["zero"],
                vllm_ctrl["scrambled"], vllm_ctrl["wrong"], vllm_ctrl["wrong_zero"], eligible,
            )

        # 3. Live vLLM Focus-1 generation on exact sealed 32 state_split JPEGs to measure full deployment speedup
        f1_dir = case_dir / "followup" / "focus" / "native" / "state_split"
        f1_rec = json.loads((f1_dir / "outputs" / "focus_state_split.json").read_bytes())
        f1_images = []
        for img_meta in f1_rec["images"]:
            with Image.open(f1_dir / img_meta["path"]) as im:
                f1_images.append(im.convert("RGB"))
        f1_out = engine.generate_vlm(f1_rec["messages"], f1_images, max_tokens=2048)

        hf_addrs = hf_audit["addresses"]
        vl_addrs = list(vllm_dec.addresses) if (vllm_dec and vllm_dec.accepted) else []
        parity = _compare_scan(hf_audit["scan"], vllm_scan)
        parity_b1 = _compare_scan(hf_audit["scan"], vllm_scan_b1)
        rows_out.append(dict(
            ordinal=ordinal,
            cid=cid,
            hf_grid=list(hf_grid),
            vllm_grid=list(vl_grid),
            grid_match=(hf_grid == vl_grid),
            hf_source_shared_s=round(float(src_shared_rec.get("generation_seconds", 0.0) + src_shared_rec.get("preprocess_seconds", 0.0)), 3),
            vllm_source_shared_s=src_out["seconds"],
            vllm_source_shared_tokens=src_out["output_tokens"],
            hf_source_shared_tokens=src_shared_rec["output_tokens"],
            receiver_scan_s=vllm_scan["seconds"],
            receiver_scan_b1_s=vllm_scan_b1["seconds"],
            receiver_ctrl_s=vllm_ctrl["seconds"] if vllm_ctrl else 0.0,
            receiver_total_s=round(vllm_scan["seconds"] + (vllm_ctrl["seconds"] if vllm_ctrl else 0.0), 3),
            hf_focus1_s=round(float(f1_rec.get("generation_seconds", 0.0) + f1_rec.get("preprocess_seconds", 0.0)), 3),
            vllm_focus1_s=f1_out["seconds"],
            hf_focus1_tokens=f1_rec["output_tokens"],
            vllm_focus1_tokens=f1_out["output_tokens"],
            hf_addresses=hf_addrs,
            vllm_addresses=vl_addrs,
            exact_address_match=(hf_addrs == vl_addrs),
            top1_match=(bool(hf_addrs) and bool(vl_addrs) and hf_addrs[0] == vl_addrs[0]),
            parity=parity,
            parity_b1=parity_b1,
        ))
        print(json.dumps(rows_out[-1], sort_keys=True), flush=True)

    summary = dict(
        status="vllm_lave_parity_bench_complete",
        model_path=str(resolved_model),
        boot_seconds=boot_s,
        batch_size=batch_size,
        mamba_cache_mode=mamba_cache_mode,
        enforce_eager=enforce_eager,
        cases=len(rows_out),
        grid_matches=sum(r["grid_match"] for r in rows_out),
        exact_address_matches=sum(r["exact_address_match"] for r in rows_out),
        top1_matches=sum(r["top1_match"] for r in rows_out),
        mean_hf_source_shared_s=round(sum(r["hf_source_shared_s"] for r in rows_out) / len(rows_out), 2),
        mean_vllm_source_shared_s=round(sum(r["vllm_source_shared_s"] for r in rows_out) / len(rows_out), 2),
        mean_receiver_total_s=round(sum(r["receiver_total_s"] for r in rows_out) / len(rows_out), 2),
        mean_receiver_scan_b1_s=round(sum(r["receiver_scan_b1_s"] for r in rows_out) / len(rows_out), 2),
        mean_hf_focus1_s=round(sum(r["hf_focus1_s"] for r in rows_out) / len(rows_out), 2),
        mean_vllm_focus1_s=round(sum(r["vllm_focus1_s"] for r in rows_out) / len(rows_out), 2),
        max_lift_diff=max(r["parity"]["max_lift_diff"] for r in rows_out),
        max_lift_diff_b1=max(r["parity_b1"]["max_lift_diff"] for r in rows_out),
        rows=rows_out,
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, default=None)
    p.add_argument("--ordinals", type=int, nargs="+", default=list(DEFAULT_ORDINALS))
    p.add_argument("--batch-size", type=int, default=33)
    p.add_argument("--gpu-mem-util", type=float, default=0.50)
    p.add_argument("--mamba-cache-mode", choices=("align", "none"), default="align")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    res = run_benchmark(
        tuple(args.ordinals),
        model_path=args.model_path,
        batch_size=args.batch_size,
        gpu_mem_util=args.gpu_mem_util,
        mamba_cache_mode=args.mamba_cache_mode,
        enforce_eager=args.enforce_eager,
    )
    if args.out is not None:
        args.out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
