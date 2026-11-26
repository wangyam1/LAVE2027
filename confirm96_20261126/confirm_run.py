"""Execute the frozen blind confirmation protocol on 96 sealed CG-Bench long videos."""

from collections import Counter
from contextlib import redirect_stdout
from dataclasses import asdict
import fcntl
import gc
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import signal
import sys
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

from entropy_routing_20261125 import multi_action, policy, zoom_verify_run as zr
from missing_fact_routing_20261124 import receiver
from missing_fact_routing_20261124.live_run import digest, emit, load_history, need, write
from missing_fact_routing_20261124.native_case import _historical_receipt
from missing_fact_routing_20261124.native_runtime import QA_TEMPLATE, _receipt
from missing_fact_routing_20261124.target import target_from_question

from .confirm_inputs import (
    ABI_BLOCK, N_CONFIRM, abi_case, hash_file, load_blind_confirmation_cases, plan_confirm_foils,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
PREFLIGHT = ROOT / "preflight.json"
SCORING_PROJECTION = ROOT / "scoring" / "metadata_projection_scanner.py"
ARMS = multi_action.ARMS
PLANNER_CAP = 512
SOURCE_CAP = 2048
QA_CAP = 16
MAX_INPUT_TOKENS = 20000
CODE = ("confirm_inputs.py", "confirm_run.py", "confirm_audit.py", "confirm_score.py", "protocol.md")
ENTROPY_CODE = ("policy.py", "multi_action.py", "experiment.py", "zoom_verify_run.py")
SHARED_CODE = ("receiver.py", "target.py", "live_run.py", "native_runtime.py", "native_case.py")


def files():
    own = {f"confirm/{n}": digest(HERE / n) for n in CODE}
    ent = {f"entropy/{n}": digest(ROOT / "entropy_routing_20261125" / n) for n in ENTROPY_CODE}
    shd = {f"shared/{n}": digest(ROOT / "missing_fact_routing_20261124" / n) for n in SHARED_CODE}
    return own | ent | shd


def attach_stage(native, runtime, historical, directory, model, stage, max_calls):
    """Attach the exact frozen native Worker for any stage (source/planner/focus/qa)."""
    need(stage in ("source", "planner", "focus", "qa") and type(max_calls) is int and 1 <= max_calls <= 2,
         f"Invalid stage {stage}")
    directory.mkdir(mode=0o700, exist_ok=False)
    for name, fname in (("manifest", "protocol_manifest.json"), ("preflight", "preflight.json")):
        with (directory / fname).open("xb") as f:
            f.write(historical[name])
    worker = native.Worker.__new__(native.Worker)
    worker.root = directory
    worker.p = dict(runtime.COMMON, planner_cap=PLANNER_CAP, max_generations=max_calls)
    worker.pre = json.loads(historical["preflight"])
    worker.preflight_sha = historical["preflight_pin"]
    worker.v = model
    worker.eos = list(dict.fromkeys([model.tokenizer.eos_token_id,
                                     model.tokenizer.convert_tokens_to_ids("<|im_end|>")]))
    worker.records = []
    worker.state = dict(pid=os.getpid(), model_calls=0, cap_hits=0, **{stage + "_calls": 0})
    costs = dict(calls=0, input_tokens=0, output_tokens=0, forward_calls=0, visual_encoder_calls=0,
                 preprocess_seconds=0.0, generation_seconds=0.0)
    worker.stage_costs = {stage: costs.copy()}
    worker.state.update({k: v for k, v in costs.items() if k != "calls"})
    return worker


def load_abi_modules():
    """Load the frozen native ITT & bridge dependencies without touching dev labels."""
    old, d, _ = load_history()
    dev_uids = {row.case["video_uid"] for row in d.rows}
    sys.path.insert(0, str(ABI_BLOCK))
    import run_itt_utility as native
    import itt_contract as contract
    split, triad, runtime, io = old.bridge.attach_code()
    historical = _historical_receipt(ABI_BLOCK)
    reference_focus = ABI_BLOCK / "reference_focus.py"
    need(reference_focus.is_file(), "Frozen reference_focus.py missing")
    return old, d, dev_uids, native, contract, split, triad, runtime, io, historical, reference_focus


def make_sampler(native, fps, frame_count):
    import numpy as np

    def sample(lo, hi):
        a, b, n = native.focus_indices(lo, hi, fps, frame_count, 32)
        return np.linspace(a, b - 1, n).astype(int).tolist()

    return sample


def source_messages(native, case_abi, overview_times, kind):
    content = []
    for t in overview_times:
        content.extend([{"type": "text", "text": f"{t:.1f}s"}, {"type": "image"}])
    return [{"role": "user", "content": content + [{"type": "text", "text": native.source_instruction(case_abi, kind)}]}]


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight requires hidden CUDA")
    import decord
    import numpy as np

    old, d, dev_uids, native, contract, split, _, _, _, _, _ = load_abi_modules()
    cases = load_blind_confirmation_cases(SCORING_PROJECTION, dev_uids=dev_uids)
    need(len(cases) == N_CONFIRM, "Expected 96 confirmation cases")

    s = d.s
    s.torch.set_num_threads(2)
    model_path = s.old.n.MODEL
    processor = s.prepare_native(s.AutoProcessor.from_pretrained(model_path, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(model_path, local_files_only=True))
    foils = plan_confirm_foils(cases, processor.tokenizer)

    media_meta = {}
    checked_targets = 0
    for case in cases:
        cid = case["id"]
        vpath = Path(case["video_path"])
        need(vpath.is_file() and vpath.stat().st_size == case["video_bytes"]
             and hash_file(vpath) == case["video_sha256"], f"Confirmation MP4 integrity failed: {cid}")
        vr = decord.VideoReader(str(vpath), ctx=decord.cpu(0), num_threads=1)
        n, fps = len(vr), float(vr.get_avg_fps())
        need(n > 64 and fps > 0, f"Invalid confirmation video geometry: {cid}")
        overview_indices = np.linspace(0, n - 1, 32).astype(int).tolist()
        overview_times = [i / fps for i in overview_indices]
        del vr
        media_meta[cid] = dict(
            frame_count=n, fps=fps, duration=n / fps, sha256=case["video_sha256"],
            overview_indices=overview_indices, overview_times=overview_times,
        )
        # Verify safe target extraction and receiver prompt layout with synthetic empty reports
        target = target_from_question(case["question"])
        foil = target_from_question(foils[cid]["question"])
        # Use canonical Qwen 32-frame grid shape check (1, 18, 28) -> 126 tokens/slot, just to verify template/verbalizers
        real_p, _ = receiver.render(s, processor, rope, (1, 18, 28), target, case["question"], ["ok", "ok"], MAX_INPUT_TOKENS)
        wrong_p, _ = receiver.render(s, processor, rope, (1, 18, 28), foil, case["question"], ["ok", "ok"], MAX_INPUT_TOKENS)
        need(real_p["context_sha256"] == wrong_p["context_sha256"]
             and real_p["target_sha256"] != wrong_p["target_sha256"]
             and real_p["labels"] == wrong_p["labels"], f"Receiver layout failed for {cid}")
        checked_targets += 1

    need(checked_targets == N_CONFIRM and not s.torch.cuda.is_initialized(), "Preflight incomplete or touched CUDA")
    order = [c["id"] for c in cases]
    manifest = dict(
        status="confirm96_preflight_frozen",
        split="confirmation",
        files=files(),
        order=order,
        cases=cases,
        media=media_meta,
        foils=foils,
        labels=0,
        clues=0,
        training=0,
        gpu=0,
        real_model_calls=0,
    )
    pin = write(PREFLIGHT, manifest)
    emit(status=manifest["status"], cases=len(order), verified_mp4=checked_targets, pin=pin)


def _verify_focus_receipt(directory, arm, rendered, ref, cap, expected_frames):
    record = _receipt(directory, ref, "focus", f"focus_{arm}", rendered["messages"], cap)
    grids, pixels = record["image_grid_thw"], record["pixel_values_shape"]
    mm_types = record["mm_token_type_ids"]
    patches, visual = sum(map(prod, grids)), sum(mm_types)
    need(len(grids) == len(record["images"]) == expected_frames and patches == pixels[0]
         and 0 < visual < record["input_tokens"]
         and [img["sha256"] for img in record["images"]] == rendered["jpeg_sha256"],
         "Focus visual receipt mismatch")
    return record, dict(
        frames=expected_frames, patches=patches, visual_tokens=visual,
        input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
        image_grid_thw=grids, pixel_values_shape=pixels,
        pixel_values_sha256=record["pixel_values_sha256"], record=ref,
    )


def _verify_qa_receipt(directory, arm, messages, choices, native, tokenizer, ref):
    record = _receipt(directory, ref, "qa", f"qa_{arm}", messages, QA_CAP)
    template = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(not record["images"] and record["rendered"] == template
         and record["input_ids"] == tokenizer.encode(template, add_special_tokens=False),
         "Pure-text QA input mismatch")
    answer = None if record["generation_cap"] else native.option(record["text"], choices)
    return dict(
        status="generated", answer=answer, capped=record["generation_cap"], record=ref,
        input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
    )


def run_one_case(
    d, case, media, foil_info, ordinal, work, model, deps, occupancy, sink,
):
    native, contract, split, triad, runtime, io, historical, reference_focus = deps
    import decord
    import focus_helpers as helper
    from PIL import Image
    from source_capture import capture_source

    work.mkdir(mode=0o700, exist_ok=False)
    case_abi = abi_case(case)
    fps, frame_count, duration = media["fps"], media["frame_count"], media["duration"]
    overview_indices = media["overview_indices"]
    overview_times = media["overview_times"]
    reader = decord.VideoReader(case["video_path"], ctx=decord.cpu(0), num_threads=1)
    need(len(reader) == frame_count, "Video frame count drifted")

    timings = {}

    # 1. Decode 32 overview frames & run live Source-shared (with single-pass state capture) + Source-query
    t0 = perf_counter()
    overview_frames = reader.get_batch(overview_indices).asnumpy()
    source_records = {}
    captured_states = None
    delegate = model.model.generate

    for kind, dir_name in (("shared", "source_shared"), ("query_specific", "source_query")):
        sdir = work / dir_name
        worker = attach_stage(native, runtime, historical, sdir, model, "source", 1)
        imgs = [native.asset(sdir, Image.fromarray(frame).convert("RGB")) for frame in overview_frames]
        msgs = source_messages(native, case_abi, overview_times, kind)
        collected = []

        def tap(**kwargs):
            need(not collected, "Duplicate capture during shared Source")
            with capture_source(model.model, kwargs) as obs:
                res = delegate(**kwargs)
            collected.append(obs)
            return res

        occupancy()
        with redirect_stdout(sink):
            if kind == "shared":
                with patch.object(model.model, "generate", side_effect=tap):
                    rec, ref = worker.generate(dir_name, "source", msgs, SOURCE_CAP, imgs)
            else:
                rec, ref = worker.generate(dir_name, "source", msgs, SOURCE_CAP, imgs)
        occupancy()
        need(rec["stage"] == "source" and rec["audit_passed"] and rec["visual_calls"] == 1
             and len(rec["images"]) == 32, f"Source audit failed: {dir_name}")
        if kind == "shared":
            need(len(collected) == 1 and collected[0]["verified"] and len(collected[0]["states"]) == 32,
                 "Live Source state capture failed")
            captured_states = collected[0]["states"]
        source_records[dir_name] = dict(record=rec, ref=ref)
    del overview_frames
    timings["source_s"] = round(perf_counter() - t0, 3)

    src_shared = source_records["source_shared"]["record"]
    src_query = source_records["source_query"]["record"]
    need(src_shared["pixel_values_sha256"] == src_query["pixel_values_sha256"]
         and src_shared["image_grid_thw"] == src_query["image_grid_thw"], "Source visual signatures diverged")
    reports = [src_shared["text"], src_query["text"]]

    # 2. Pure-text live Planner
    t0 = perf_counter()
    pdir = work / "plan"
    pworker = attach_stage(native, runtime, historical, pdir, model, "planner", 1)
    pmessages_all, ptools = contract.planning_messages(case_abi, reports, duration, overview_times)
    pmessages = pmessages_all["T_plus"]
    occupancy()
    with redirect_stdout(sink):
        prec, pref = pworker.generate("live_text_plan", "planner", pmessages, PLANNER_CAP, None, ptools)
    occupancy()
    need(prec["stage"] == "planner" and prec["audit_passed"] and prec["visual_calls"] == 0 and not prec["images"],
         "Planner audit failed")
    parsed_plan = contract.plan_record(prec, duration, media)
    text_action = parsed_plan["arguments"] if parsed_plan["valid"] else None
    timings["planner_s"] = round(perf_counter() - t0, 3)

    # 3. Entropy-guided calibrated-lift receiver (batch_size=1, 0 pixels, 3 strict content controls)
    t0 = perf_counter()
    target = target_from_question(case["question"])
    wrong_target = target_from_question(foil_info["question"])
    sample = make_sampler(native, fps, frame_count)
    geometry = dict(source=overview_indices, fps=fps, count=frame_count)
    row_stub = SimpleNamespace(
        cid=case["id"], seed_index=case["seed_index"],
        requests=[dict(max_input_tokens=MAX_INPUT_TOKENS)],
    )

    def choose(eligible):
        grid, = {tuple(sz) for sz in src_shared["image_grid_thw"]}
        rope = d.s.rope(model.model.config)
        actual = receiver.render(d.s, model.processor, rope, grid, target, text_action["query"], reports, MAX_INPUT_TOKENS)
        occupancy()
        scan = receiver.probe_states(d, model, captured_states, actual)
        occupancy()
        candidate = policy.nominate(scan["real"], scan["zero"], eligible)
        audit = dict(scan=scan, controls=None, decision=None, addresses=[], per_gap_frames=0,
                     actual_prompt=actual[0], wrong_prompt=None)
        if candidate is not None:
            foil_layout = receiver.render(d.s, model.processor, rope, grid, wrong_target, text_action["query"], reports, MAX_INPUT_TOKENS)
            occupancy()
            control = receiver.probe_controls(d, row_stub, model, captured_states, candidate.slot, actual, foil_layout)
            occupancy()
            decision = policy.approve(candidate, scan["real"], scan["zero"], control["scrambled"],
                                      control["wrong"], control["wrong_zero"], eligible)
            audit.update(
                controls=control,
                decision=dict(asdict(decision), addresses=list(decision.addresses)),
                wrong_prompt=foil_layout[0],
                addresses=list(decision.addresses) if decision.accepted else [],
                per_gap_frames=decision.per_gap_frames if decision.accepted else 0,
            )
        return audit

    from entropy_routing_20261125.experiment import plan_actions
    action, sampler, route_audit = plan_actions(split, text_action, geometry, sample, choose)
    del captured_states
    timings["receiver_s"] = round(perf_counter() - t0, 3)

    # 4. Focus-1 (100% unmodified native Focus prompt, 32 frames/arm)
    t0 = perf_counter()
    f1_visual_ledger = {}
    f1_budget = None

    def observe_f1(arm, indices, query):
        fdir = work / "focus1" / arm
        fdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        worker = attach_stage(native, runtime, historical, fdir, model, "focus", 1)
        rendered = triad.render(native, io, helper, reference_focus, fdir, query, indices, reader, fps, frame_count)
        occupancy()
        with redirect_stdout(sink):
            obs = triad.one_focus(worker, arm, rendered)
        occupancy()
        _, vinfo = _verify_focus_receipt(fdir, arm, rendered, obs["record"], SOURCE_CAP, len(indices))
        nonlocal f1_budget
        bud = (vinfo["frames"], vinfo["patches"], vinfo["visual_tokens"])
        if f1_budget is None:
            f1_budget = bud
        need(bud == f1_budget, "Focus-1 visual budget differed across arms")
        f1_visual_ledger[arm] = vinfo
        return obs

    f1_ledger = multi_action.execute(action, ordinal, sampler, text_action, observe_f1)
    timings["focus1_s"] = round(perf_counter() - t0, 3)

    # 5. Pure-text QA-1 (exact-input deduplicated)
    t0 = perf_counter()
    arms_order = multi_action.order_for(ordinal)
    qa1_cells = {}
    qa_cache = {}
    unique_qa1 = 0
    for arm in arms_order:
        row_f1 = f1_ledger["rows"][arm]
        obs1 = row_f1["observation"]
        if obs1 is None:
            qa1_cells[arm] = dict(status="common_invalid", observation=None, capped=False, qa=None, qa_reused_from=None)
            continue
        msgs = io.qa_messages(native, case_abi, reports, obs1)
        mkey = json.dumps(msgs, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if mkey in qa_cache:
            origin, qa_rec = qa_cache[mkey]
            qa1_cells[arm] = dict(status="native", observation=obs1, capped=row_f1["capped"],
                                  qa=qa_rec, qa_reused_from=origin)
        else:
            qdir = work / "qa1" / arm
            qdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            qworker = attach_stage(native, runtime, historical, qdir, model, "qa", 1)
            occupancy()
            with redirect_stdout(sink):
                _, qref = qworker.generate(f"qa_{arm}", "qa", msgs, QA_CAP)
            occupancy()
            qa_rec = _verify_qa_receipt(qdir, arm, msgs, case["choices"], native, model.tokenizer, qref)
            qa_cache[mkey] = (dict(stage="qa1", arm=arm), qa_rec)
            qa1_cells[arm] = dict(status="native", observation=obs1, capped=row_f1["capped"],
                                  qa=qa_rec, qa_reused_from=None)
            unique_qa1 += 1
    timings["qa1_s"] = round(perf_counter() - t0, 3)

    # 6. Focus-2 dense 16-frame verification (100% unmodified native Focus prompt & original text_action["query"])
    t0 = perf_counter()
    row_for_zoom = SimpleNamespace(media=media)
    prior_stub = dict(
        text_action=text_action,
        routes=action,
        cells={arm: dict(native=qa1_cells[arm]) for arm in ARMS},
    )
    f2_plan = zr.plan_case_zoom(row_for_zoom, prior_stub)
    f2_focus = {}
    f2_visual_ledger = {}
    f2_budget = None
    phys_f2 = 0

    if f2_plan["triggered"]:
        seen_views = {}
        for arm in arms_order:
            indices = f2_plan["arms"][arm]["indices"]
            vkey = tuple(indices)
            if vkey in seen_views:
                src_arm = seen_views[vkey]
                f2_focus[arm] = dict(f2_focus[src_arm], calls=0, reused_from=src_arm)
                continue
            qobj = dict(query=text_action["query"], start_time=indices[0] / fps, end_time=(indices[-1] + 1) / fps)
            f2dir = work / "focus2" / arm
            f2dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            worker = attach_stage(native, runtime, historical, f2dir, model, "focus", 1)
            rendered = triad.render(native, io, helper, reference_focus, f2dir, qobj, indices, reader, fps, frame_count)
            occupancy()
            with redirect_stdout(sink):
                obs = triad.one_focus(worker, arm, rendered)
            occupancy()
            _, vinfo = _verify_focus_receipt(f2dir, arm, rendered, obs["record"], SOURCE_CAP, zr.ZOOM_FRAMES)
            bud = (vinfo["frames"], vinfo["patches"], vinfo["visual_tokens"])
            if f2_budget is None:
                f2_budget = bud
            need(bud == f2_budget, "Focus-2 visual budget differed across arms")
            f2_focus[arm] = dict(obs, calls=1)
            f2_visual_ledger[arm] = vinfo
            seen_views[vkey] = arm
            phys_f2 += 1
    timings["focus2_s"] = round(perf_counter() - t0, 3)

    # 7. Pure-text QA-2 (reusing QA-1 whenever combined observation is unchanged!)
    t0 = perf_counter()
    qa2_cells = {}
    unique_qa2 = 0
    for arm in arms_order:
        base_ans = (qa1_cells[arm]["qa"] or {}).get("answer")
        base_obs = qa1_cells[arm]["observation"]
        kind = f2_plan["arms"][arm]["kind"]
        if not f2_plan["triggered"]:
            qa2_cells[arm] = dict(
                kind=kind, equal_budget_answer=base_ans, on_policy_answer=base_ans,
                observation=base_obs, f2_observation=None, f2_capped=False, qa=None, qa_reused_from=None,
            )
            continue
        frow = f2_focus[arm]
        combined = zr.combine_observation(base_obs, frow["observation"], frow["capped"])
        msgs = io.qa_messages(native, case_abi, reports, combined)
        mkey = json.dumps(msgs, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if mkey in qa_cache:
            origin, qa_rec = qa_cache[mkey]
            reused_from = origin
        else:
            qdir = work / "qa2" / arm
            qdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            qworker = attach_stage(native, runtime, historical, qdir, model, "qa", 1)
            occupancy()
            with redirect_stdout(sink):
                _, qref = qworker.generate(f"qa_{arm}", "qa", msgs, QA_CAP)
            occupancy()
            qa_rec = _verify_qa_receipt(qdir, arm, msgs, case["choices"], native, model.tokenizer, qref)
            qa_cache[mkey] = (dict(stage="qa2", arm=arm), qa_rec)
            reused_from = None
            unique_qa2 += 1
        eq_ans = qa_rec["answer"]
        op_ans = eq_ans if kind == "on_policy" else base_ans
        qa2_cells[arm] = dict(
            kind=kind, equal_budget_answer=eq_ans, on_policy_answer=op_ans,
            observation=combined, f2_observation=frow["observation"], f2_capped=frow["capped"],
            qa=qa_rec, qa_reused_from=reused_from,
        )
    timings["qa2_s"] = round(perf_counter() - t0, 3)

    del reader
    gc.collect()

    return dict(
        cid=case["id"],
        rank=case["rank"],
        ordinal=ordinal,
        qid=case["qid"],
        case_uid=case["video_uid"],
        split="confirmation",
        foil_id=foil_info["foil_id"],
        target=asdict(target),
        sources=dict(source_shared=source_records["source_shared"]["ref"],
                     source_query=source_records["source_query"]["ref"]),
        planner_receipt=pref,
        text_action=text_action,
        routes=action,
        route_audit=route_audit,
        focus1=f1_ledger,
        focus1_visual_ledger=f1_visual_ledger,
        qa1=qa1_cells,
        unique_qa1_calls=unique_qa1,
        focus2_plan=f2_plan,
        focus2=f2_focus,
        focus2_visual_ledger=f2_visual_ledger,
        physical_focus2_calls=phys_f2,
        qa2=qa2_cells,
        unique_qa2_calls=unique_qa2,
        timings=timings,
    )


def run(pin, uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose one exclusive A40")
    need(digest(PREFLIGHT) == pin, "Changed preflight receipt")
    manifest = json.loads(PREFLIGHT.read_bytes())
    need(manifest["status"] == "confirm96_preflight_frozen" and manifest["files"] == files()
         and manifest["labels"] == manifest["clues"] == manifest["training"] == manifest["gpu"] == 0
         and len(manifest["cases"]) == N_CONFIRM, "Changed frozen confirmation preflight")

    old, d, dev_uids, native, contract, split, triad, runtime, io, historical, reference_focus = load_abi_modules()
    need(not ({c["video_uid"] for c in manifest["cases"]} & dev_uids), "Confirmation overlaps dev!")
    deps = (native, contract, split, triad, runtime, io, historical, reference_focus)

    t = d.s.torch
    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, util = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        need(name == "NVIDIA A40" and int(free) >= 36864 and int(util) <= 15
             and not old.smi(uuid, "--query-compute-apps=pid"), "Selected A40 is in use")
        d.s.old.verify_weights()
        need(t.cuda.device_count() == 1 and not t.cuda.memory_reserved() and t.cuda.get_allocator_backend() == "native",
             "Unexpected CUDA context")
        t.cuda.set_per_process_memory_fraction(old.CAP / t.cuda.get_device_properties(0).total_memory)
        peak = 0

        def occupancy():
            nonlocal peak
            peak = max(peak, t.cuda.max_memory_reserved())
            need(set(old.smi(uuid, "--query-compute-apps=pid").split()) <= {str(os.getpid())}
                 and peak <= old.CAP, "GPU ownership or allocation cap violated")

        def stop(*_):
            raise InterruptedError("Confirmation 96 run stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700, exist_ok=False)
        totals, committed = Counter(), {}
        terminal = dict(
            split="confirmation", order=manifest["order"], pin=pin, gpu=uuid, cards=1,
            labels=0, clues=0, training=0, pid=os.getpid(),
        )
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP, cohort=N_CONFIRM)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                live_foils = plan_confirm_foils(manifest["cases"], model.processor.tokenizer)
                need(live_foils == manifest["foils"], "CPU and live tokenizer disagree on confirmation foils")
                versions = tuple(p._version for p in model.model.parameters())
                for ordinal, case in enumerate(manifest["cases"]):
                    cid = case["id"]
                    work = ACTION / f"case_{ordinal:02d}"
                    emit(status="case_start", ordinal=ordinal, cid=cid, cases=len(committed))
                    t_case = perf_counter()
                    rec = run_one_case(
                        d, case, manifest["media"][cid], live_foils[cid], ordinal, work, model, deps, occupancy, sink,
                    )
                    rec["elapsed_seconds"] = round(perf_counter() - t_case, 3)
                    committed[cid] = write(work / "case.json", rec)
                    trace = rec["route_audit"]
                    k_used = len(trace["addresses"]) if trace and trace["addresses"] else 0
                    totals.update(
                        source_calls=2, planner_calls=1,
                        invalid_action=int(rec["text_action"] is None),
                        scan=int(trace is not None),
                        approved=int(k_used > 0),
                        focus1_calls=rec["focus1"]["physical_calls"],
                        qa1_calls=rec["unique_qa1_calls"],
                        f2_triggered=int(rec["focus2_plan"]["triggered"]),
                        focus2_calls=rec["physical_focus2_calls"],
                        qa2_calls=rec["unique_qa2_calls"],
                        **({f"k_{k_used}": 1} if k_used > 0 else {}),
                    )
                    emit(status="case_committed", ordinal=ordinal, cid=cid, cases=len(committed),
                         k_used=k_used, f2=int(rec["focus2_plan"]["triggered"]),
                         elapsed_s=rec["elapsed_seconds"], peak_bytes=peak, **totals)
                need(tuple(p._version for p in model.model.parameters()) == versions
                     and len(committed) == N_CONFIRM, "Model weights or cohort changed")
                occupancy()
        except BaseException as error:
            write(ACTION / "terminal.json", dict(status="stopped", error=type(error).__name__, cases=len(committed),
                                                rows=committed, counts=totals, peak_bytes=peak, **terminal))
            raise
        write(ACTION / "terminal.json", dict(status="complete", cases=len(committed), rows=committed, counts=totals,
                                            peak_bytes=peak, **terminal))
        emit(status="inference_complete", cases=len(committed), peak_bytes=peak, **totals)


if __name__ == "__main__":
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "preflight" and not args:
        preflight()
    elif mode == "run" and len(args) == 2:
        run(*args)
    else:
        raise SystemExit("preflight | run PIN UUID")
