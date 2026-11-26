"""Benchmark batched receiver throughput, VRAM, and exact decision fidelity vs batch_size=1."""

from contextlib import redirect_stdout
import fcntl
import json
from math import isfinite
import os
from pathlib import Path
import sys
from time import perf_counter

from missing_fact_routing_20261124 import receiver
from missing_fact_routing_20261124.foil_plan import plan_foils
from missing_fact_routing_20261124.live_run import load_history, need
from missing_fact_routing_20261124.target import target_from_question

from . import multi_action, policy


def evaluate_batched(d, vlm, groups, batch_size):
    """Exact same embedding-only no-pixel contract as receiver._evaluate, using microbatches of size batch_size."""
    torch, dev = d.s.torch, d.s.DEV
    model = vlm.model

    def only_embeddings(_module, _args, kwargs):
        if kwargs.get("inputs_embeds") is None or any(
            kwargs.get(k) is not None for k in ("input_ids", "pixel_values", "pixel_values_videos")
        ):
            raise ValueError("Receiver requires embeddings without pixel or token-id inputs")

    def forbid_vision(*_):
        raise ValueError("Receiver invoked its vision encoder")

    prior_rope = model.model.rope_deltas
    model.model.rope_deltas = None
    hooks = []
    results = []
    try:
        hooks.append(model.register_forward_pre_hook(only_embeddings, with_kwargs=True))
        hooks.append(model.model.visual.register_forward_pre_hook(forbid_vision))
        with torch.inference_mode():
            for layout, states in groups:
                _, (ids, mask, mm, positions, slots, labels) = layout
                base = model.get_input_embeddings()(ids.to(dev))
                mask_d, mm_d, pos_d = (x.to(dev) for x in (mask, mm, positions))
                states = list(states)
                if len(labels) != 2 or labels[0] == labels[1] or base[0, slots].shape != states[0].shape:
                    raise ValueError("The receiver slot or Y/N labels do not match the Source")
                pairs = []
                for start in range(0, len(states), batch_size):
                    chunk = states[start:start + batch_size]
                    bsz = len(chunk)
                    embeds = base.expand(bsz, -1, -1).clone()
                    for j, st in enumerate(chunk):
                        embeds[j, slots] = st
                    out = model(
                        inputs_embeds=embeds,
                        attention_mask=mask_d.expand(bsz, -1),
                        mm_token_type_ids=mm_d.expand(bsz, -1),
                        position_ids=pos_d.expand(-1, bsz, -1),
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    logits = out.logits[:, -1, labels].float().tolist()
                    for pair in logits:
                        if not all(map(isfinite, pair)):
                            raise ValueError("Nonfinite batched receiver logits")
                        pairs.append(tuple(pair))
                    del embeds, out
                results.append(pairs)
    finally:
        for hook in hooks:
            hook.remove()
        model.model.rope_deltas = prior_rope
    return results


def main():
    uuid = sys.argv[1]
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose one exclusive A40")
    old, d, order = load_history()
    import run_itt_utility as native
    t = d.s.torch
    root = Path("/data1/llj/lljmas/qwen35_cgbench_entropy_routing_20261125_01")
    work_tmp = root / "bench_batched_tmp"
    work_tmp.mkdir(mode=0o700, exist_ok=True)

    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, util = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        need(name == "NVIDIA A40" and int(free) >= 36864 and int(util) <= 15
             and not old.smi(uuid, "--query-compute-apps=pid"), "Selected A40 is in use")
        t.cuda.set_per_process_memory_fraction(old.CAP / t.cuda.get_device_properties(0).total_memory)
        peak = 0

        def occupancy():
            nonlocal peak
            peak = max(peak, t.cuda.max_memory_reserved())
            need(set(old.smi(uuid, "--query-compute-apps=pid").split()) <= {str(os.getpid())}
                 and peak <= old.CAP, "GPU cap exceeded")

        with (work_tmp / "native.log").open("w", buffering=1) as sink:
            with redirect_stdout(sink):
                model = d.s.boot(work_tmp, uuid, d.nat)
            occupancy()
            foils = plan_foils(d.rows, model.processor.tokenizer)
            rope = d.s.rope(model.model.config)
            # 4 representative exposed dev videos (2 causal wins + 2 shared-window wins)
            sample_ordinals = (16, 6, 20, 38)
            batch_sizes = (1, 4, 8, 11)
            report = []
            for idx in sample_ordinals:
                row = d.rows[idx]
                case_dir = work_tmp / f"case_{idx:02d}"
                if case_dir.exists():
                    import shutil
                    shutil.rmtree(case_dir)
                case_dir.mkdir(mode=0o700)
                reports, states = old.live_sources(d, row, model, case_dir, occupancy, sink)
                prior_case = json.loads((root / "action" / f"case_{idx:02d}" / "case.json").read_text())
                text = prior_case["text_action"]
                geo, sample, _ = old.bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
                base = sample(text["start_time"], text["end_time"])
                _, menu = multi_action.gap_pools(geo["source"], geo["fps"], base, sample)
                eligible = sorted(menu)
                grid, = {tuple(sz) for sz in reports[0]["image_grid_thw"]}
                cap = row.requests[0].get("max_input_tokens", 20000)
                history = [r["text"] for r in reports]
                target = target_from_question(row.case["question"])
                foil_target = target_from_question(foils[row.cid]["question"])
                actual = receiver.render(d.s, model.processor, rope, grid, target, text["query"], history, cap)
                wrong = receiver.render(d.s, model.processor, rope, grid, foil_target, text["query"], history, cap)
                nil = t.zeros_like(states[0])

                # Reference batch_size=1
                per_bsz = {}
                ref_scan = None
                ref_dec = None
                for bsz in batch_sizes:
                    t.cuda.reset_peak_memory_stats()
                    t.cuda.synchronize()
                    t0 = perf_counter()
                    (scan_scores,) = evaluate_batched(d, model, ((actual, (nil, *states)),), bsz)
                    zero_s, real_s = scan_scores[0], scan_scores[1:]
                    nom = policy.nominate(real_s, zero_s, eligible)
                    ctrl_scores = None
                    dec = None
                    if nom is not None:
                        active = states[nom.slot]
                        rng = t.Generator().manual_seed(16423 + row.seed_index)
                        channels = t.randperm(active.shape[1], generator=rng).to(active.device)
                        signs = (t.randint(0, 2, (active.shape[1],), generator=rng) * 2 - 1).to(active.device, dtype=active.dtype)
                        perturbed = (active[:, channels] * signs).contiguous()
                        scram_list, (wz, wr) = evaluate_batched(
                            d, model, ((actual, (perturbed,)), (wrong, (nil, active))), bsz,
                        )
                        dec = policy.approve(nom, real_s, zero_s, scram_list[0], wr, wz, eligible)
                        ctrl_scores = (scram_list[0], wz, wr)
                    t.cuda.synchronize()
                    dt = perf_counter() - t0
                    bsz_peak = t.cuda.max_memory_reserved()
                    occupancy()
                    if bsz == 1:
                        ref_scan = scan_scores
                        ref_dec = dec
                        max_diff = 0.0
                        exact_match = True
                    else:
                        max_diff = max(abs(a[j] - b[j]) for a, b in zip(ref_scan, scan_scores) for j in (0, 1))
                        exact_match = (
                            (dec is None) == (ref_dec is None)
                            and (dec is None or (
                                dec.accepted == ref_dec.accepted
                                and dec.addresses == ref_dec.addresses
                                and dec.per_gap_frames == ref_dec.per_gap_frames
                                and dec.nomination.slot == ref_dec.nomination.slot
                                and dec.nomination.address == ref_dec.nomination.address
                            ))
                        )
                    per_bsz[str(bsz)] = dict(
                        seconds=round(dt, 3),
                        peak_gib=round(bsz_peak / (1024**3), 3),
                        max_logit_diff=round(max_diff, 6),
                        exact_decision_match=exact_match,
                        accepted=None if dec is None else dec.accepted,
                        addresses=[] if dec is None else list(dec.addresses),
                        tokens=actual[0]["tokens"],
                    )
                report.append(dict(ordinal=idx, cid=row.cid, tokens=actual[0]["tokens"], results=per_bsz))
                del states
                t.cuda.empty_cache()
        print(json.dumps(dict(status="batched_receiver_benchmark_complete", cases=report), indent=2))


if __name__ == "__main__":
    main()
