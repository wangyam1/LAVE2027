"""High-throughput dual-GPU vLLM-LAVE runner for 150 category-balanced LVB samples.

Uses full A40 VRAM (gpu_memory_utilization=0.90) + cross-video / cross-arm continuous
batching (4 videos x 3 arms = up to 12 concurrent VLM generations per batch) while
preserving 100% exact single-pass Source visual-state capture, zero-vision Receiver
guard, exact frozen LAVE routing policy, and atomic resume across already-committed cases.
"""

from bisect import bisect_left
from collections import Counter
from dataclasses import asdict
import gc
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from time import perf_counter
from types import SimpleNamespace

from entropy_routing_20261125 import multi_action, policy, zoom_verify_run as zr
from missing_fact_routing_20261124.target import target_from_question
from vllm_lave_20261126.engine import VLLMLaveEngine, render_receiver_vllm_prompt

from .lvb_inputs import (
    N_SAMPLES,
    VideoFrameReader,
    abi_case,
    load_blind_lvb_cases,
    plan_lvb_foils,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
PREFLIGHT = ROOT / "preflight.json"
MODEL_PATH = Path("/data1/llj/development/lljvideo/models/Qwen3.5-9B")
NATIVE_MIN_PIXELS = 65536
NATIVE_MAX_PIXELS = 262144
ARMS = multi_action.ARMS
SOURCE_CAP = 2048
PLANNER_CAP = 512
QA_CAP = 16
BATCH_CASES = 4


def need(ok, msg):
    if not ok:
        raise ValueError(msg)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def emit(**rec):
    print(json.dumps(rec, sort_keys=True, allow_nan=False), flush=True)


def write(path, rec):
    path = Path(path)
    scratch = path.with_name(path.name + ".pending")
    need(not path.exists(), f"Output already exists: {path}")
    with scratch.open("x") as f:
        json.dump(rec, f, sort_keys=True, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    scratch.rename(path)
    return digest(path)


def source_messages(acq, case_abi, overview_times, kind):
    content = []
    for t in overview_times:
        content.extend([{"type": "text", "text": f"{t:.1f}s"}, {"type": "image"}])
    return [{"role": "user", "content": content + [{"type": "text", "text": acq.source_instruction(case_abi, kind)}]}]


def focus_messages(indices, fps, query):
    """Exact native Focus prompt envelope matching reference_focus.py / native_joint_bridge.py."""
    content = [{"type": "text", "text": "Sampled video frames (timestamps below):\n"}]
    for idx in indices:
        content.append({"type": "text", "text": f"{idx / fps:.1f}s"})
        content.append({"type": "image"})
    content.append({
        "type": "text",
        "text": (
            f"Question:\n{query}\n\n"
            "Please answer the question based on the given video clip. "
            "For every relevant visual fact, begin a separate line with "
            "the exact provided frame timestamp in the form '12.3s: ...'. "
            "Use only timestamps shown above. "
            "If the clip is not related to the question, please return 'No relevant content found.'"
        ),
    })
    return [{"role": "user", "content": content}]


def direct_visual_qa_messages(indices, fps, case, reports, f1_obs=None):
    """Diagnostic VLM QA directly over the exact retrieved Focus frames + Source text reports."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(case["choices"]))
    content = [{"type": "text", "text": "Sampled video frames (timestamps below):\n"}]
    for idx in indices:
        content.append({"type": "text", "text": f"{idx / fps:.1f}s"})
        content.append({"type": "image"})
    extra = f"Initial clip notes:\n{f1_obs}\n\n" if f1_obs else ""
    body = (
        f"Overview report 1:\n{reports[0]}\n\n"
        f"Overview report 2:\n{reports[1]}\n\n"
        f"{extra}"
        f"Question:\n{case['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Answer using the sampled video frames above and overview reports. "
        "Reply with ONLY the single option letter (e.g., A)."
    )
    content.append({"type": "text", "text": body})
    return [{"role": "user", "content": content}]


def make_sampler(focus_indices_fn, fps, frame_count):
    import numpy as np

    def sample(lo, hi):
        a, b, n = focus_indices_fn(lo, hi, fps, frame_count, 32)
        return np.linspace(a, b - 1, n).astype(int).tolist()

    return sample


def distance_sorted(frame, seen):
    """Exact distance from a single integer frame index to the nearest anchor in sorted ``seen``."""
    pos = bisect_left(seen, frame)
    best = None
    if pos < len(seen):
        best = abs(seen[pos] - frame)
    if pos > 0:
        cand = abs(frame - seen[pos - 1])
        best = cand if best is None else min(best, cand)
    return 0 if best is None else best


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight requires hidden CUDA")
    import numpy as np
    from transformers import AutoTokenizer

    cases = load_blind_lvb_cases(n_total=N_SAMPLES)
    need(len(cases) == N_SAMPLES, f"Expected {N_SAMPLES} cases")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), local_files_only=True)
    foils = plan_lvb_foils(cases, tokenizer)

    media_meta = {}
    unique_vid_cache = {}
    for case in cases:
        cid = case["id"]
        vp = case["video_path"]
        if vp not in unique_vid_cache:
            reader = VideoFrameReader(vp)
            try:
                n, fps = reader.frame_count, reader.fps
                need(n > 64 and fps > 0, f"Invalid LVB geometry: {vp}")
                ov_idx = np.linspace(0, n - 1, 32).astype(int).tolist()
                probe = reader.read_pil_frames([ov_idx[0], ov_idx[15], ov_idx[-1]])
                need(len(probe) == 3, f"Decode probe failed: {vp}")
                unique_vid_cache[vp] = dict(
                    frame_count=n, fps=fps, duration=n / fps,
                    overview_indices=ov_idx, overview_times=[k / fps for k in ov_idx],
                )
            finally:
                reader.close()
        media_meta[cid] = unique_vid_cache[vp]
        target = target_from_question(case["question"])
        foil = target_from_question(foils[cid]["question"])
        p1, l1 = render_receiver_vllm_prompt(tokenizer, target, case["question"], ["ok", "ok"])
        p2, l2 = render_receiver_vllm_prompt(tokenizer, foil, case["question"], ["ok", "ok"])
        need(l1 == l2 and p1 != p2, f"Receiver prompt verification failed: {cid}")

    cat_counts = Counter(c["question_category"] for c in cases)
    dur_counts = Counter(c["duration_group"] for c in cases)
    unique_vids = len({c["video_path"] for c in cases})
    manifest = dict(
        status="lvb150_vllm_preflight_frozen",
        split="lvb_eval",
        n=len(cases),
        unique_videos=unique_vids,
        category_counts=dict(sorted(cat_counts.items())),
        duration_groups=dict(sorted(dur_counts.items())),
        order=[c["id"] for c in cases],
        cases=cases,
        media=media_meta,
        foils=foils,
        labels=0,
        training=0,
        gpu=0,
    )
    pin = write(PREFLIGHT, manifest)
    emit(
        status=manifest["status"],
        cases=len(cases),
        unique_videos=unique_vids,
        categories=len(cat_counts),
        cat_min=min(cat_counts.values()),
        cat_max=max(cat_counts.values()),
        duration_groups=manifest["duration_groups"],
        pin=pin,
    )


def run_batch(engine, batch_items, manifest, acq, contract, focus_indices_fn):
    """Execute a micro-batch of up to 4 cases together via cross-video/cross-arm vLLM batching."""
    from entropy_routing_20261125.experiment import plan_actions

    n_b = len(batch_items)
    t_batch_start = perf_counter()
    states = []
    for ordinal, case in batch_items:
        cid = case["id"]
        media = manifest["media"][cid]
        states.append(dict(
            ordinal=ordinal,
            case=case,
            case_abi=abi_case(case),
            media=media,
            foil_info=manifest["foils"][cid],
            reader=VideoFrameReader(case["video_path"]),
            timings={},
        ))

    try:
        # 1a. Source_shared 1-by-1 (guarantees exact unmixed 32-slot post-merger capture per video)
        t0 = perf_counter()
        ov_imgs_batch = []
        for st in states:
            imgs = st["reader"].read_pil_frames(st["media"]["overview_indices"])
            ov_imgs_batch.append(imgs)
            t_s = perf_counter()
            s_out = engine.generate_source_shared(
                source_messages(acq, st["case_abi"], st["media"]["overview_times"], "shared"),
                imgs, max_tokens=SOURCE_CAP,
            )
            st["src_shared"] = s_out
            st["grid"], = set(s_out["grids"])
            st["captured_slots"] = s_out["slots"]
            st["timings"]["source_shared_s"] = round(perf_counter() - t_s, 3)

        # 1b. Source_query batched across all cases in micro-batch (hits MM cache!)
        t_sq = perf_counter()
        sq_reqs = [
            (source_messages(acq, st["case_abi"], st["media"]["overview_times"], "query_specific"), imgs)
            for st, imgs in zip(states, ov_imgs_batch)
        ]
        sq_outs = engine.generate_vlm_batch(sq_reqs, max_tokens=SOURCE_CAP)
        sq_per = round((perf_counter() - t_sq) / n_b, 3)
        for st, sq in zip(states, sq_outs):
            st["reports"] = [st["src_shared"]["text"], sq["text"]]
            st["timings"]["source_s"] = round(st["timings"]["source_shared_s"] + sq_per, 3)
        del ov_imgs_batch, sq_reqs

        # 2. Pure-text Planner batched across all cases in micro-batch
        t_pl = perf_counter()
        p_msgs = []
        ptools = None
        for st in states:
            pm_all, ptools = contract.planning_messages(
                st["case_abi"], st["reports"], st["media"]["duration"], st["media"]["overview_times"],
            )
            p_msgs.append(pm_all["T_plus"])
        p_outs = engine.generate_text_batch(p_msgs, max_tokens=PLANNER_CAP, tools=ptools)
        pl_per = round((perf_counter() - t_pl) / n_b, 3)
        for st, pout in zip(states, p_outs):
            parsed = contract.plan_record(
                dict(text=pout["text"], generation_cap=pout["capped"]), st["media"]["duration"], st["media"],
            )
            st["planner_raw"] = pout["text"]
            st["planner_error"] = parsed.get("error")
            st["text_action"] = parsed["arguments"] if parsed["valid"] else None
            st["timings"]["planner_s"] = pl_per

        # 3. Zero-pixel Receiver (already batch=33 prefix-cached, ~3.5s/case!) + frozen multi-gap routing
        split_stub = SimpleNamespace(distance_sorted=distance_sorted)
        for st in states:
            t_r = perf_counter()
            case, media = st["case"], st["media"]
            target = target_from_question(case["question"])
            wrong_target = target_from_question(st["foil_info"]["question"])
            sample = make_sampler(focus_indices_fn, media["fps"], media["frame_count"])
            geometry = dict(source=media["overview_indices"], fps=media["fps"], count=media["frame_count"])

            def choose(eligible, _st=st, _t=target, _wt=wrong_target):
                scan = engine.probe_states(
                    _t, _st["text_action"]["query"], _st["reports"], _st["grid"], _st["captured_slots"], batch_size=33,
                )
                need(scan["forwards"] == 33 and scan["visual_forwards"] == 0, "Receiver scan invariant violated")
                cand = policy.nominate(scan["real"], scan["zero"], eligible)
                audit = dict(
                    scan=dict(real=scan["real"], zero=scan["zero"], forwards=33, visual_forwards=0),
                    controls=None, decision=None, addresses=[], per_gap_frames=0,
                )
                if cand is not None:
                    ctrl = engine.probe_controls(
                        _st["case"]["seed_index"], _t, _wt, _st["text_action"]["query"],
                        _st["reports"], _st["grid"], _st["captured_slots"], cand.slot,
                    )
                    need(ctrl["forwards"] == 3 and ctrl["visual_forwards"] == 0, "Receiver control invariant violated")
                    dec = policy.approve(
                        cand, scan["real"], scan["zero"],
                        ctrl["scrambled"], ctrl["wrong"], ctrl["wrong_zero"], eligible,
                    )
                    audit.update(
                        controls=dict(
                            scrambled=ctrl["scrambled"], wrong=ctrl["wrong"], wrong_zero=ctrl["wrong_zero"],
                            forwards=3, visual_forwards=0,
                        ),
                        decision=dict(asdict(dec), addresses=list(dec.addresses)),
                        addresses=list(dec.addresses) if dec.accepted else [],
                        per_gap_frames=dec.per_gap_frames if dec.accepted else 0,
                    )
                return audit

            action, _, route_audit = plan_actions(split_stub, st["text_action"], geometry, sample, choose)
            st["routes"] = action
            st["route_audit"] = route_audit
            del st["captured_slots"]
            st["timings"]["receiver_s"] = round(perf_counter() - t_r, 3)

        # 4. Focus-1 (up to 4x3=12 VLM calls in ONE batch!) + DirectVQA-1 (up to 12 calls in ONE batch!)
        t_f1 = perf_counter()
        f1_jobs = []   # (st_idx, arm, indices, imgs)
        for si, st in enumerate(states):
            st["f1_rows"] = {}
            st["dvqa1"] = {}
            st["phys_f1"] = 0
            seen = {}
            for arm in multi_action.order_for(st["ordinal"]):
                indices = st["routes"]["indices"][arm]
                if not indices:
                    st["f1_rows"][arm] = dict(status="common_invalid", calls=0, indices=[], observation=None, capped=False)
                    st["dvqa1"][arm] = None
                    continue
                vkey = tuple(indices)
                if vkey in seen:
                    st["f1_rows"][arm] = dict(_reuse_from=seen[vkey])
                else:
                    imgs = st["reader"].read_pil_frames(indices)
                    f1_jobs.append((si, arm, indices, imgs))
                    seen[vkey] = arm
                    st["phys_f1"] += 1

        if f1_jobs:
            f1_reqs = [
                (focus_messages(ind, states[si]["media"]["fps"], states[si]["text_action"]["query"]), imgs)
                for si, _, ind, imgs in f1_jobs
            ]
            f1_outs = engine.generate_vlm_batch(f1_reqs, max_tokens=SOURCE_CAP)
            dv1_reqs = [
                (direct_visual_qa_messages(ind, states[si]["media"]["fps"], states[si]["case"], states[si]["reports"]), imgs)
                for si, _, ind, imgs in f1_jobs
            ]
            dv1_outs = engine.generate_vlm_batch(dv1_reqs, max_tokens=QA_CAP)
            for (si, arm, ind, _), fout, dvout in zip(f1_jobs, f1_outs, dv1_outs):
                st = states[si]
                st["f1_rows"][arm] = dict(
                    status="focus_generated", calls=1, indices=ind,
                    observation=fout["text"], capped=fout["capped"], output_tokens=fout["output_tokens"],
                )
                st["dvqa1"][arm] = None if dvout["capped"] else acq.option(dvout["text"], st["case"]["choices"])
            del f1_reqs, dv1_reqs
        del f1_jobs
        for st in states:
            for arm in multi_action.order_for(st["ordinal"]):
                if "_reuse_from" in st["f1_rows"][arm]:
                    src = st["f1_rows"][arm]["_reuse_from"]
                    st["f1_rows"][arm] = dict(st["f1_rows"][src], calls=0, reused_from=src)
                    st["dvqa1"][arm] = st["dvqa1"][src]
        f1_per = round((perf_counter() - t_f1) / n_b, 3)
        for st in states:
            st["timings"]["focus1_s"] = f1_per

        # 5. Pure-text QA-1 batched across all cases & unique observations in micro-batch
        t_q1 = perf_counter()
        qa1_jobs = []   # (st_idx, mkey, msgs)
        for si, st in enumerate(states):
            st["qa_cache"] = {}
            st["unique_qa1"] = 0
            for arm in multi_action.order_for(st["ordinal"]):
                obs1 = st["f1_rows"][arm]["observation"]
                if obs1 is None:
                    continue
                msgs = acq.final_messages(st["case_abi"], [*st["reports"], obs1])
                mkey = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
                if mkey not in st["qa_cache"]:
                    st["qa_cache"][mkey] = None
                    qa1_jobs.append((si, mkey, msgs))
                    st["unique_qa1"] += 1
        if qa1_jobs:
            q1_outs = engine.generate_text_batch([m for _, _, m in qa1_jobs], max_tokens=QA_CAP)
            for (si, mkey, _), qout in zip(qa1_jobs, q1_outs):
                ans = None if qout["capped"] else acq.option(qout["text"], states[si]["case"]["choices"])
                states[si]["qa_cache"][mkey] = dict(answer=ans, text=qout["text"], capped=qout["capped"])
        for st in states:
            st["qa1_cells"] = {}
            for arm in multi_action.order_for(st["ordinal"]):
                obs1 = st["f1_rows"][arm]["observation"]
                if obs1 is None:
                    st["qa1_cells"][arm] = dict(status="common_invalid", observation=None, capped=False, qa=None)
                else:
                    msgs = acq.final_messages(st["case_abi"], [*st["reports"], obs1])
                    mkey = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
                    st["qa1_cells"][arm] = dict(
                        status="native", observation=obs1, capped=st["f1_rows"][arm]["capped"], qa=st["qa_cache"][mkey],
                    )
        q1_per = round((perf_counter() - t_q1) / n_b, 3)
        for st in states:
            st["timings"]["qa1_s"] = q1_per

        # 6. Focus-2 (16 dense frames, up to 12 VLM calls in ONE batch!) + DirectVQA-2 (up to 12 calls in ONE batch!)
        t_f2 = perf_counter()
        f2_jobs = []
        for si, st in enumerate(states):
            st["f2_plan"] = zr.plan_case_zoom(
                SimpleNamespace(media=st["media"]),
                dict(text_action=st["text_action"], routes=st["routes"],
                     cells={a: dict(native=st["qa1_cells"][a]) for a in ARMS}),
            )
            st["f2_rows"] = {}
            st["dvqa2_eq"] = {}
            st["phys_f2"] = 0
            if st["f2_plan"]["triggered"]:
                seen = {}
                for arm in multi_action.order_for(st["ordinal"]):
                    indices = st["f2_plan"]["arms"][arm]["indices"]
                    vkey = tuple(indices)
                    if vkey in seen:
                        st["f2_rows"][arm] = dict(_reuse_from=seen[vkey])
                    else:
                        imgs = st["reader"].read_pil_frames(indices)
                        f2_jobs.append((si, arm, indices, imgs))
                        seen[vkey] = arm
                        st["phys_f2"] += 1

        if f2_jobs:
            f2_reqs = [
                (focus_messages(ind, states[si]["media"]["fps"], states[si]["text_action"]["query"]), imgs)
                for si, _, ind, imgs in f2_jobs
            ]
            f2_outs = engine.generate_vlm_batch(f2_reqs, max_tokens=SOURCE_CAP)
            dv2_reqs = [
                (direct_visual_qa_messages(
                    ind, states[si]["media"]["fps"], states[si]["case"], states[si]["reports"],
                    states[si]["qa1_cells"][arm]["observation"],
                ), imgs)
                for si, arm, ind, imgs in f2_jobs
            ]
            dv2_outs = engine.generate_vlm_batch(dv2_reqs, max_tokens=QA_CAP)
            for (si, arm, ind, _), fout, dvout in zip(f2_jobs, f2_outs, dv2_outs):
                st = states[si]
                st["f2_rows"][arm] = dict(
                    status="focus_generated", calls=1, indices=ind,
                    observation=fout["text"], capped=fout["capped"], output_tokens=fout["output_tokens"],
                )
                st["dvqa2_eq"][arm] = None if dvout["capped"] else acq.option(dvout["text"], st["case"]["choices"])
            del f2_reqs, dv2_reqs
        del f2_jobs

        for st in states:
            st["dvqa2"] = {}
            if not st["f2_plan"]["triggered"]:
                for arm in ARMS:
                    st["dvqa2"][arm] = dict(on_policy=st["dvqa1"][arm], equal_budget=st["dvqa1"][arm])
            else:
                for arm in multi_action.order_for(st["ordinal"]):
                    if "_reuse_from" in st["f2_rows"][arm]:
                        src = st["f2_rows"][arm]["_reuse_from"]
                        st["f2_rows"][arm] = dict(st["f2_rows"][src], calls=0, reused_from=src)
                        st["dvqa2_eq"][arm] = st["dvqa2_eq"][src]
                    kind = st["f2_plan"]["arms"][arm]["kind"]
                    eq = st["dvqa2_eq"][arm]
                    st["dvqa2"][arm] = dict(
                        equal_budget=eq,
                        on_policy=eq if kind == "on_policy" else st["dvqa1"][arm],
                    )
        f2_per = round((perf_counter() - t_f2) / n_b, 3)
        for st in states:
            st["timings"]["focus2_s"] = f2_per

        # 7. Pure-text QA-2 batched across all triggered cases in micro-batch
        t_q2 = perf_counter()
        qa2_jobs = []
        for si, st in enumerate(states):
            st["unique_qa2"] = 0
            st["comb_obs"] = {}
            if not st["f2_plan"]["triggered"]:
                continue
            for arm in multi_action.order_for(st["ordinal"]):
                base_obs = st["qa1_cells"][arm]["observation"]
                frow = st["f2_rows"][arm]
                comb = zr.combine_observation(base_obs, frow["observation"], frow["capped"])
                st["comb_obs"][arm] = comb
                msgs = acq.final_messages(st["case_abi"], [*st["reports"], comb])
                mkey = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
                if mkey not in st["qa_cache"]:
                    st["qa_cache"][mkey] = None
                    qa2_jobs.append((si, mkey, msgs))
                    st["unique_qa2"] += 1
        if qa2_jobs:
            q2_outs = engine.generate_text_batch([m for _, _, m in qa2_jobs], max_tokens=QA_CAP)
            for (si, mkey, _), qout in zip(qa2_jobs, q2_outs):
                ans = None if qout["capped"] else acq.option(qout["text"], states[si]["case"]["choices"])
                states[si]["qa_cache"][mkey] = dict(answer=ans, text=qout["text"], capped=qout["capped"])

        q2_per = round((perf_counter() - t_q2) / n_b, 3)
        elapsed_per = round((perf_counter() - t_batch_start) / n_b, 3)
        records = []
        for st in states:
            st["timings"]["qa2_s"] = q2_per
            qa2_cells = {}
            for arm in multi_action.order_for(st["ordinal"]):
                base_ans = (st["qa1_cells"][arm]["qa"] or {}).get("answer")
                base_obs = st["qa1_cells"][arm]["observation"]
                kind = st["f2_plan"]["arms"][arm]["kind"]
                if not st["f2_plan"]["triggered"]:
                    qa2_cells[arm] = dict(
                        kind=kind, equal_budget_answer=base_ans, on_policy_answer=base_ans,
                        observation=base_obs, f2_observation=None, f2_capped=False, qa=None,
                    )
                else:
                    comb = st["comb_obs"][arm]
                    frow = st["f2_rows"][arm]
                    msgs = acq.final_messages(st["case_abi"], [*st["reports"], comb])
                    mkey = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
                    qrec = st["qa_cache"][mkey]
                    eq_ans = qrec["answer"]
                    qa2_cells[arm] = dict(
                        kind=kind, equal_budget_answer=eq_ans,
                        on_policy_answer=eq_ans if kind == "on_policy" else base_ans,
                        observation=comb, f2_observation=frow["observation"], f2_capped=frow["capped"],
                        qa=qrec,
                    )
            case = st["case"]
            records.append(dict(
                cid=case["id"],
                ordinal=st["ordinal"],
                video_uid=case["video_uid"],
                question_category=case["question_category"],
                duration_group=case["duration_group"],
                duration=round(st["media"]["duration"], 2),
                grid=list(st["grid"]),
                foil_id=st["foil_info"]["foil_id"],
                text_action=st["text_action"],
                planner_raw=st["planner_raw"],
                planner_error=st["planner_error"],
                source_vision_calls=st["src_shared"]["vision_calls"],
                routes=st["routes"],
                route_audit=st["route_audit"],
                focus1=dict(rows=st["f1_rows"], physical_calls=st["phys_f1"]),
                qa1=st["qa1_cells"],
                unique_qa1_calls=st["unique_qa1"],
                focus2_plan=st["f2_plan"],
                focus2=dict(rows=st["f2_rows"], physical_calls=st["phys_f2"]),
                qa2=qa2_cells,
                unique_qa2_calls=st["unique_qa2"],
                direct_visual_qa=dict(focus1=st["dvqa1"], focus2=st["dvqa2"]),
                timings=st["timings"],
                elapsed_seconds=elapsed_per,
            ))
        return records
    finally:
        for st in states:
            st["reader"].close()
        gc.collect()


def run_shard(pin, shard_id, num_shards, gpu_idx):
    shard_id, num_shards = int(shard_id), int(num_shards)
    need(0 <= shard_id < num_shards and os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu_idx),
         "Bad shard or GPU binding")
    need(digest(PREFLIGHT) == pin, "Preflight pin mismatch")
    manifest = json.loads(PREFLIGHT.read_bytes())
    need(manifest["status"] == "lvb150_vllm_preflight_frozen" and manifest["labels"] == 0, "Invalid manifest")

    sys.path.insert(0, str(ROOT / "contracts"))
    import acquisition_contract as acq
    import baseline_common as bcom
    import itt_contract as contract

    def stop(*_):
        raise InterruptedError(f"LVB shard {shard_id} stopped")

    signal.signal(signal.SIGTERM, stop)
    ACTION.mkdir(mode=0o700, exist_ok=True)
    shard_cases = [(i, c) for i, c in enumerate(manifest["cases"]) if i % num_shards == shard_id]

    # Resume cleanly over already-committed cases (preserve completed work, clean partial dirs)
    totals, committed, pending = Counter(), {}, []
    for ordinal, case in shard_cases:
        cid = case["id"]
        cdir = ACTION / f"case_{ordinal:03d}"
        cfile = cdir / "case.json"
        if cfile.is_file():
            rec = json.loads(cfile.read_bytes())
            need(rec["cid"] == cid and rec["ordinal"] == ordinal, f"Corrupt resumed case: {cid}")
            committed[cid] = digest(cfile)
            trace = rec["route_audit"]
            k_used = len(trace["addresses"]) if trace and trace["addresses"] else 0
            totals.update(
                done=1,
                invalid_action=int(rec["text_action"] is None),
                scan=int(trace is not None),
                approved=int(k_used > 0),
                f1_calls=rec["focus1"]["physical_calls"],
                f2_triggered=int(rec["focus2_plan"]["triggered"]),
                f2_calls=rec["focus2"]["physical_calls"],
                **({f"k_{k_used}": 1} if k_used > 0 else {}),
            )
        else:
            if cdir.exists():
                shutil.rmtree(cdir)
            pending.append((ordinal, case))

    emit(status="shard_boot", shard=shard_id, num_shards=num_shards, gpu_idx=gpu_idx,
         pid=os.getpid(), assigned=len(shard_cases), resumed=len(committed), pending=len(pending),
         gpu_mem_util=0.90, batch_cases=BATCH_CASES)

    term_path = ACTION / f"shard_{shard_id}.terminal.json"
    if term_path.exists():
        term_path.unlink()

    if pending:
        t_boot = perf_counter()
        engine = VLLMLaveEngine(
            MODEL_PATH,
            max_model_len=20000,
            max_num_batched_tokens=32768,
            gpu_memory_utilization=0.90,
            max_num_seqs=64,
            enable_prefix_caching=True,
            mamba_cache_mode="align",
            enforce_eager=False,
            mm_processor_kwargs=dict(size={"shortest_edge": NATIVE_MIN_PIXELS, "longest_edge": NATIVE_MAX_PIXELS}),
        )
        emit(status="shard_engine_ready", shard=shard_id, boot_s=round(perf_counter() - t_boot, 2))
        try:
            for start in range(0, len(pending), BATCH_CASES):
                chunk = pending[start:start + BATCH_CASES]
                recs = run_batch(engine, chunk, manifest, acq, contract, bcom.focus_indices)
                for rec in recs:
                    ordinal, cid = rec["ordinal"], rec["cid"]
                    cdir = ACTION / f"case_{ordinal:03d}"
                    cdir.mkdir(mode=0o700, exist_ok=False)
                    committed[cid] = write(cdir / "case.json", rec)
                    trace = rec["route_audit"]
                    k_used = len(trace["addresses"]) if trace and trace["addresses"] else 0
                    totals.update(
                        done=1,
                        invalid_action=int(rec["text_action"] is None),
                        scan=int(trace is not None),
                        approved=int(k_used > 0),
                        f1_calls=rec["focus1"]["physical_calls"],
                        f2_triggered=int(rec["focus2_plan"]["triggered"]),
                        f2_calls=rec["focus2"]["physical_calls"],
                        **({f"k_{k_used}": 1} if k_used > 0 else {}),
                    )
                    emit(
                        status="case_committed", shard=shard_id, ordinal=ordinal, cid=cid,
                        cat=rec["question_category"], k_used=k_used, f2=int(rec["focus2_plan"]["triggered"]),
                        elapsed_s=rec["elapsed_seconds"], receiver_s=rec["timings"]["receiver_s"], **totals,
                    )
        except BaseException as err:
            write(term_path, dict(
                status="stopped", shard=shard_id, error=type(err).__name__, detail=str(err),
                cases=len(committed), rows=committed, counts=totals, pid=os.getpid(), gpu_idx=gpu_idx, pin=pin,
            ))
            raise

    write(term_path, dict(
        status="complete", shard=shard_id, num_shards=num_shards, assigned=len(shard_cases),
        cases=len(committed), rows=committed, counts=totals, pid=os.getpid(), gpu_idx=gpu_idx, pin=pin,
        labels=0, training=0,
    ))
    emit(status="shard_complete", shard=shard_id, cases=len(committed), **totals)


if __name__ == "__main__":
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "preflight" and not args:
        preflight()
    elif mode == "run" and len(args) == 4:
        run_shard(*args)
    else:
        raise SystemExit("preflight | run PIN SHARD_ID NUM_SHARDS GPU_IDX")
