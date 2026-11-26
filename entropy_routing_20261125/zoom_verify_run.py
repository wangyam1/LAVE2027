"""Focus-2 dense 16-frame verification built on the audited entropy_routing_20261125 run."""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
from hashlib import sha256
import json
from math import prod
import os
from pathlib import Path
import re
import signal
import sys
from time import perf_counter

from missing_fact_routing_20261124.live_run import digest, emit, load_history, need, write
from missing_fact_routing_20261124.native_case import _historical_receipt
from missing_fact_routing_20261124.native_runtime import QA_TEMPLATE, _receipt

from . import multi_action


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
PREFLIGHT = ROOT / "preflight.json"
PRIOR = Path("/data1/llj/lljmas/qwen35_cgbench_entropy_routing_20261125_01")
PRIOR_RELEASE = "ecefe26c7b0ca388cb8b733be2e626e359c174b0fa2e90c499c9bc6cf57f0ec7"
PRIOR_SCORE = "b50529df1df171489cdcdc979ac0fa68c01dd3268ed616e66d6c62eac346de6b"
ARMS = multi_action.ARMS
MODES = ("native_query", "question_query")
ZOOM_FRAMES = 16
CLUSTER_SPAN_S = 30.0
CLUSTER_PAD_S = 10.0
MIN_WINDOW_S = 20.0
NO_CONTENT = "No relevant content found."
TS_RE = re.compile(r"(\d+(?:\.\d+)?)s\s*:")
CODE = ("zoom_verify_run.py", "zoom_verify_score.py", "multi_action.py")


def read(path, pin=None):
    if pin is not None:
        need(digest(path) == pin, f"Changed receipt: {Path(path).name}")
    return json.loads(Path(path).read_bytes())


def code():
    return {name: digest(HERE / name) for name in CODE}


def seal(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    return sha256(raw).hexdigest()


def parse_focus1_timestamps(observation, duration):
    """Extract deterministic reported timestamps from non-empty native Focus-1 lines."""
    if not isinstance(observation, str):
        return []
    found = []
    for line in observation.splitlines():
        if NO_CONTENT in line:
            continue
        match = TS_RE.search(line)
        if match:
            value = float(match.group(1))
            if 0.0 <= value <= duration:
                found.append(value)
    return sorted(set(found))


def densest_cluster_window(timestamps, duration, *, span=CLUSTER_SPAN_S, pad=CLUSTER_PAD_S):
    """Pick the densest <=span-sec cluster of F1 timestamps (ties: tighter, then earlier), padded by +-pad."""
    pts = sorted(set(t for t in timestamps if 0.0 <= t <= duration))
    if not pts:
        return None
    best = None
    for i, lo in enumerate(pts):
        j = i
        while j + 1 < len(pts) and pts[j + 1] - lo <= span:
            j += 1
        rank = (j - i + 1, -(pts[j] - lo), -lo)
        if best is None or rank > best[0]:
            best = (rank, lo, pts[j])
    _, lo, hi = best
    start = max(0.0, lo - pad)
    end = min(duration, hi + pad)
    if end - start < MIN_WINDOW_S:
        mid = 0.5 * (start + end)
        start = max(0.0, mid - MIN_WINDOW_S / 2.0)
        end = min(duration, start + MIN_WINDOW_S)
        start = max(0.0, end - MIN_WINDOW_S)
    return start, end


def pick_unseen_frames(lo_s, hi_s, *, fps, frame_count, seen, k=ZOOM_FRAMES):
    """Sample exactly k chronological unseen frame indices inside [lo_s, hi_s]."""
    lo_i = max(0, int(round(lo_s * fps)))
    hi_i = min(frame_count - 1, int(round(hi_s * fps)))
    pool = [idx for idx in range(lo_i, hi_i + 1) if idx not in seen]
    if len(pool) < k:
        return None
    return multi_action.evenly_native(pool, k)


def plan_case_zoom(row, prior_case):
    """Build deterministic, label-free Focus-2 16-frame plans (on-policy + equal-budget fallback)."""
    text_action = prior_case["text_action"]
    if text_action is None:
        return dict(triggered=False, arms={a: dict(kind="common_invalid", indices=[]) for a in ARMS})

    duration = row.media["duration"]
    fps = row.media["fps"]
    frame_count = row.media["frame_count"]
    source_set = set(row.media["overview_indices"])

    plans = {}
    any_on_policy = False
    for arm in ARMS:
        obs = prior_case["cells"][arm]["native"]["observation"] or ""
        f1_indices = prior_case["routes"]["indices"][arm]
        seen = source_set | set(f1_indices)
        ts = parse_focus1_timestamps(obs, duration)
        window = densest_cluster_window(ts, duration)
        frames = None
        if window is not None:
            frames = pick_unseen_frames(window[0], window[1], fps=fps, frame_count=frame_count, seen=seen)
        if frames is not None:
            any_on_policy = True
            plans[arm] = dict(kind="on_policy", timestamps=ts, window=list(window), indices=frames)
        else:
            plans[arm] = dict(kind="no_on_policy", timestamps=ts, window=None, indices=[])

    if not any_on_policy:
        return dict(triggered=False, arms=plans)

    # Equal-budget fallback for arms that did not produce Focus-1 timestamps on a triggered video:
    # sample 16 unseen frames evenly inside their own text_action window (or full video if exhausted).
    for arm in ARMS:
        if plans[arm]["kind"] == "on_policy":
            continue
        seen = source_set | set(prior_case["routes"]["indices"][arm])
        lo, hi = text_action["start_time"], text_action["end_time"]
        frames = pick_unseen_frames(lo, hi, fps=fps, frame_count=frame_count, seen=seen)
        fallback_window = [lo, hi]
        if frames is None:
            frames = pick_unseen_frames(0.0, duration, fps=fps, frame_count=frame_count, seen=seen)
            fallback_window = [0.0, duration]
        need(frames is not None and len(frames) == ZOOM_FRAMES, "Could not construct equal-budget Focus-2 frames")
        plans[arm] = dict(kind="equal_budget_fallback", timestamps=plans[arm]["timestamps"],
                          window=fallback_window, indices=frames)

    return dict(triggered=True, arms=plans)


def combine_observation(f1_obs, f2_obs, f2_capped):
    """Combine native Focus-1 report with dense Focus-2 verification cleanly."""
    clean_f1 = (f1_obs or "").strip()
    clean_f2 = (f2_obs or "").strip()
    if f2_capped or not clean_f2 or clean_f2 == NO_CONTENT:
        return f1_obs
    if not clean_f1 or clean_f1 == NO_CONTENT:
        return clean_f2
    return clean_f1 + "\n\nFocused dense inspection:\n" + clean_f2


def load_resources():
    old, d, order = load_history()
    import run_itt_utility as native

    _, triad, runtime, io = old.bridge.attach_code()
    release = read(PRIOR / "release.json", PRIOR_RELEASE)
    terminal = read(PRIOR / "action" / "terminal.json", release["terminal"])
    need(release["status"] == "entropy_routing_factorial_audited" and release["gpu_released"] is True
         and terminal["status"] == "complete" and terminal["order"] == order
         and terminal["labels"] == terminal["holdout"] == terminal["training"] == 0,
         "Changed parent entropy_routing run")
    items = []
    for i, row in enumerate(d.rows):
        prior_work = PRIOR / "action" / f"case_{i:02d}"
        prior_case = read(prior_work / "case.json", terminal["rows"][row.cid])
        need(prior_case["cid"] == row.cid and prior_case["ordinal"] == i
             and prior_case["case_uid"] == row.case["video_uid"], "Moved parent case")
        reports = [
            read(prior_work / kind / "outputs" / f"{kind}.json")["text"]
            for kind in ("source_shared", "source_query")
        ]
        plan = plan_case_zoom(row, prior_case)
        items.append((row, prior_case, reports, plan))
    return old, d, order, native, triad, runtime, io, items


def roster(items):
    return {row.cid: plan for row, _, _, plan in items}


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for Focus-2 preflight")
    _, d, order, _, _, _, _, items = load_resources()
    plans = roster(items)
    triggered = sum(p["triggered"] for p in plans.values())
    counts = Counter(p["arms"][a]["kind"] for p in plans.values() for a in ARMS)
    need(triggered > 0 and not d.s.torch.cuda.is_initialized(), "Invalid Focus-2 preflight")
    manifest = dict(
        status="zoom_verify_inputs_frozen", code=code(), order=order, parent=PRIOR_RELEASE,
        roster_sha256=seal(plans), triggered_cases=triggered, arm_kinds=dict(counts),
        labels=0, holdout=0, gpu=0,
    )
    pin = write(PREFLIGHT, manifest)
    emit(status=manifest["status"], cases=len(order), triggered=triggered, arm_kinds=dict(counts), pin=pin)


def query_for_mode(mode, row, prior_case, indices, fps):
    bounds = dict(start_time=indices[0] / fps, end_time=(indices[-1] + 1) / fps)
    if mode == "native_query":
        return dict(query=prior_case["text_action"]["query"], **bounds)
    if mode == "question_query":
        return dict(query=row.case["question"], **bounds)
    raise ValueError(f"Unknown Focus-2 mode: {mode}")


def _verify_focus_receipt(directory, arm, proof_render, ref, cap):
    record = _receipt(directory, ref, "focus", f"focus_{arm}", proof_render["messages"], cap)
    grids, pixels = record["image_grid_thw"], record["pixel_values_shape"]
    mm_types = record["mm_token_type_ids"]
    patches, visual_tokens = sum(map(prod, grids)), sum(mm_types)
    need(len(grids) == len(record["images"]) == ZOOM_FRAMES and patches == pixels[0]
         and 0 < visual_tokens < record["input_tokens"]
         and [img["sha256"] for img in record["images"]] == proof_render["jpeg_sha256"],
         "Focus-2 visual receipt mismatch")
    return record, dict(
        frames=ZOOM_FRAMES, patches=patches, visual_tokens=visual_tokens,
        input_tokens=record["input_tokens"], output_tokens=record["output_tokens"],
        image_grid_thw=grids, pixel_values_shape=pixels,
        pixel_values_sha256=record["pixel_values_sha256"], record=ref,
    )


def _verify_qa_receipt(work, mode, arm, row, reports, observation, native, runtime, io, tokenizer, ref):
    directory = work / "qa" / mode / arm
    messages = io.qa_messages(native, row.case, reports, observation)
    record = _receipt(directory, ref, "qa", f"qa_{arm}", messages, runtime.COMMON["qa_cap"])
    template = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(not record["images"] and record["rendered"] == template
         and record["input_ids"] == tokenizer.encode(template, add_special_tokens=False),
         "Focus-2 pure-text QA input changed")
    answer = None if record["generation_cap"] else native.option(record["text"], row.case["choices"])
    return dict(status="generated", answer=answer, capped=record["generation_cap"], record=ref,
                input_tokens=record["input_tokens"], output_tokens=record["output_tokens"])


def run_case(resources, item, ordinal, work, model, occupancy, sink):
    old, d, _, native, triad, runtime, io, _ = resources
    row, prior_case, reports, plan = item
    arms_order = multi_action.order_for(ordinal)
    modes_order = MODES if (ordinal // 6) % 2 == 0 else MODES[::-1]

    cells = {}
    for arm in ARMS:
        base_cell = prior_case["cells"][arm]["native"]
        base_ans = (base_cell.get("qa") or {}).get("answer")
        base_obs = base_cell.get("observation")
        cells[arm] = dict(
            kind=plan["arms"][arm]["kind"],
            baseline_answer=base_ans,
            baseline_observation=base_obs,
            modes={},
        )

    if not plan["triggered"]:
        for arm in ARMS:
            for mode in MODES:
                cells[arm]["modes"][mode] = dict(
                    equal_budget_answer=cells[arm]["baseline_answer"],
                    on_policy_answer=cells[arm]["baseline_answer"],
                    observation=cells[arm]["baseline_observation"],
                    f2_observation=None, f2_capped=False, qa=None, qa_reused_from=None,
                )
        return dict(cid=row.cid, ordinal=ordinal, case_uid=row.case["video_uid"],
                    plan=plan, focus={}, visual_ledger={}, cells=cells,
                    physical_focus_calls=0, unique_qa_calls=0)

    import decord
    import focus_helpers as helper
    from live_dev import media_path

    geo, _, reference = old.bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
    hist = _historical_receipt(row.job["block"])
    reader = decord.VideoReader(str(media_path(row)), ctx=decord.cpu(0), num_threads=1)
    cap = runtime.COMMON["source_cap"]

    focus_ledger = {mode: {} for mode in MODES}
    visual_ledger = {mode: {} for mode in MODES}
    visual_signatures = {}
    visual_budget = None
    physical_focus_calls = 0

    for mode in modes_order:
        seen_views = {}
        for arm in arms_order:
            indices = plan["arms"][arm]["indices"]
            vkey = tuple(indices)
            if vkey in seen_views:
                src_arm = seen_views[vkey]
                focus_ledger[mode][arm] = dict(focus_ledger[mode][src_arm], calls=0, reused_from=src_arm)
                continue
            qobj = query_for_mode(mode, row, prior_case, indices, geo["fps"])
            directory = work / "focus" / mode / arm
            directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            worker = runtime.attach(native, hist, directory, model, "focus", 1)
            rendered = triad.render(native, io, helper, reference, directory, qobj, indices,
                                    reader, geo["fps"], geo["count"])
            occupancy()
            with redirect_stdout(sink):
                result = triad.one_focus(worker, arm, rendered)
            occupancy()
            _, vinfo = _verify_focus_receipt(directory, arm, rendered, result["record"], cap)
            bud = (vinfo["frames"], vinfo["patches"], vinfo["visual_tokens"])
            if visual_budget is None:
                visual_budget = bud
            need(bud == visual_budget, "Focus-2 visual budget differed across arms/modes")
            sig = (rendered["clocks"], rendered["jpeg_sha256"], rendered["rgb_sha256"],
                   vinfo["image_grid_thw"], vinfo["pixel_values_shape"], vinfo["pixel_values_sha256"])
            need(visual_signatures.setdefault(vkey, sig) == sig, "Matched Focus-2 visual input changed")
            focus_ledger[mode][arm] = dict(result, calls=1)
            visual_ledger[mode][arm] = vinfo
            seen_views[vkey] = arm
            physical_focus_calls += 1

    # Pure-text QA with exact-input deduplication (also reusing baseline native QA if observation unchanged)
    previous_qa = {}
    for arm in arms_order:
        base_obs = cells[arm]["baseline_observation"]
        base_ans = cells[arm]["baseline_answer"]
        if isinstance(base_obs, str):
            bkey = json.dumps(io.qa_messages(native, row.case, reports, base_obs),
                              ensure_ascii=False, sort_keys=True, allow_nan=False)
            previous_qa.setdefault(bkey, (dict(kind="prior_baseline", arm=arm), dict(status="reused_baseline", answer=base_ans)))

    unique_qa_calls = 0
    for arm in arms_order:
        for mode in modes_order:
            frow = focus_ledger[mode][arm]
            f1_obs = cells[arm]["baseline_observation"]
            combined = combine_observation(f1_obs, frow["observation"], frow["capped"])
            messages = io.qa_messages(native, row.case, reports, combined)
            mkey = json.dumps(messages, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if mkey in previous_qa:
                origin_info, qa_rec = previous_qa[mkey]
                reused_from = origin_info
            else:
                directory = work / "qa" / mode / arm
                directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                worker = runtime.attach(native, hist, directory, model, "qa", 1)
                occupancy()
                with redirect_stdout(sink):
                    _, ref = worker.generate(f"qa_{arm}", "qa", messages, runtime.COMMON["qa_cap"])
                occupancy()
                qa_rec = _verify_qa_receipt(work, mode, arm, row, reports, combined,
                                            native, runtime, io, model.tokenizer, ref)
                previous_qa[mkey] = (dict(kind="generated", mode=mode, arm=arm), qa_rec)
                reused_from = None
                unique_qa_calls += 1
            eq_ans = qa_rec["answer"]
            op_ans = eq_ans if plan["arms"][arm]["kind"] == "on_policy" else cells[arm]["baseline_answer"]
            cells[arm]["modes"][mode] = dict(
                equal_budget_answer=eq_ans,
                on_policy_answer=op_ans,
                observation=combined,
                f2_observation=frow["observation"],
                f2_capped=frow["capped"],
                qa=qa_rec,
                qa_reused_from=reused_from,
            )

    return dict(
        cid=row.cid, ordinal=ordinal, case_uid=row.case["video_uid"],
        plan=plan, focus=focus_ledger, visual_ledger=visual_ledger, cells=cells,
        physical_focus_calls=physical_focus_calls, unique_qa_calls=unique_qa_calls,
    )


def run(pin, uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose one exclusive A40")
    resources = load_resources()
    old, d, order, _, _, _, _, items = resources
    manifest = read(PREFLIGHT, pin)
    need(manifest["code"] == code() and manifest["order"] == order and manifest["parent"] == PRIOR_RELEASE
         and manifest["roster_sha256"] == seal(roster(items))
         and manifest["labels"] == manifest["holdout"] == manifest["gpu"] == 0, "Changed Focus-2 manifest")
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
                 and peak <= old.CAP, "GPU occupancy or cap changed")

        def stop(*_):
            raise InterruptedError("Focus-2 verification stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700)
        done, counts = {}, Counter()
        terminal = dict(order=order, pin=pin, parent=PRIOR_RELEASE, cards=1, gpu=uuid, pid=os.getpid(),
                        labels=0, holdout=0, training=0)
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                versions = tuple(p._version for p in model.model.parameters())
                for i, item in enumerate(items):
                    row = item[0]
                    work = ACTION / f"case_{i:02d}"
                    work.mkdir(mode=0o700)
                    start = perf_counter()
                    rec = run_case(resources, item, i, work, model, occupancy, sink)
                    rec["elapsed_seconds"] = perf_counter() - start
                    done[row.cid] = write(work / "case.json", rec)
                    counts.update(
                        triggered=int(rec["plan"]["triggered"]),
                        focus_calls=rec["physical_focus_calls"],
                        unique_qa=rec["unique_qa_calls"],
                    )
                    emit(status="case_committed", ordinal=i, cases=len(done), peak_bytes=peak, **counts)
                need(tuple(p._version for p in model.model.parameters()) == versions, "Model weights changed")
                occupancy()
        except BaseException as error:
            write(ACTION / "terminal.json", dict(status="stopped", error=type(error).__name__, cases=len(done),
                                                rows=done, counts=counts, peak_bytes=peak, **terminal))
            raise
        write(ACTION / "terminal.json", dict(status="complete", cases=len(done), rows=done, counts=counts,
                                            peak_bytes=peak, **terminal))
        emit(status="inference_complete", cases=len(done), peak_bytes=peak, **counts)


def audit():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for CPU release audit")
    resources = load_resources()
    old, d, order, native, _, runtime, io, items = resources
    manifest = read(PREFLIGHT)
    need(manifest["code"] == code() and manifest["order"] == order and manifest["parent"] == PRIOR_RELEASE
         and manifest["roster_sha256"] == seal(roster(items)), "Changed Focus-2 design")
    tpath = ACTION / "terminal.json"
    terminal = read(tpath)
    pid = str(terminal["pid"])
    proc_stat = Path("/proc") / pid / "stat"
    need(terminal["status"] == "complete" and terminal["order"] == order and terminal["cases"] == len(order)
         and terminal["pin"] == digest(PREFLIGHT) and terminal["labels"] == terminal["holdout"] == 0
         and (not proc_stat.exists() or proc_stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
         and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(), "Await GPU release")

    s = d.s
    s.torch.set_num_threads(2)
    proc = s.prepare_native(s.AutoProcessor.from_pretrained(s.old.n.MODEL, local_files_only=True), d.nat)
    cap = runtime.COMMON["source_cap"]
    totals = Counter()

    for i, item in enumerate(items):
        row, prior_case, reports, plan = item
        work = ACTION / f"case_{i:02d}"
        rec = read(work / "case.json", terminal["rows"][row.cid])
        need(rec["cid"] == row.cid and rec["ordinal"] == i and rec["case_uid"] == row.case["video_uid"]
             and rec["plan"] == plan, "Case or Focus-2 plan changed")
        if not plan["triggered"]:
            need(rec["physical_focus_calls"] == 0 and rec["unique_qa_calls"] == 0, "Untriggered case spent calls")
            totals.update(cases=1)
            continue

        budgets = set()
        sigs = {}
        phys_f = 0
        for mode in MODES:
            for arm in ARMS:
                frow = rec["focus"][mode][arm]
                if frow["calls"] == 0:
                    src = frow["reused_from"]
                    need(plan["arms"][arm]["indices"] == plan["arms"][src]["indices"]
                         and frow["observation"] == rec["focus"][mode][src]["observation"], "Bad Focus-2 reuse")
                    continue
                need(frow["calls"] == 1 and frow["indices"] == plan["arms"][arm]["indices"], "Bad Focus-2 indices")
                qstr = "Question:\n" + query_for_mode(mode, row, prior_case, frow["indices"], row.media["fps"])["query"] + "\n\n"
                directory = work / "focus" / mode / arm
                ref = frow["record"]
                raw_rec = read(directory / ref["path"], ref["sha256"])
                proof = _receipt(directory, ref, "focus", f"focus_{arm}", raw_rec["messages"], cap)
                need(qstr in proof["messages"][0]["content"][-1]["text"]
                     and proof["text"] == frow["observation"] and proof["generation_cap"] == frow["capped"],
                     "Focus-2 prompt or output mismatch")
                grids, pixels = proof["image_grid_thw"], proof["pixel_values_shape"]
                patches, visual = sum(map(prod, grids)), sum(proof["mm_token_type_ids"])
                need(len(grids) == len(proof["images"]) == ZOOM_FRAMES and patches == pixels[0]
                     and [img["sha256"] for img in proof["images"]] == frow["jpeg_sha256"], "Bad Focus-2 JPEG receipt")
                for img in proof["images"]:
                    need(digest(directory / img["path"]) == img["sha256"], "Cached Focus-2 JPEG changed")
                budgets.add((ZOOM_FRAMES, patches, visual))
                vkey = tuple(frow["indices"])
                sig = (frow["clocks"], frow["jpeg_sha256"], frow["rgb_sha256"], grids, pixels, proof["pixel_values_sha256"])
                need(sigs.setdefault(vkey, sig) == sig, "Matched Focus-2 visual inputs differed across modes")
                phys_f += 1
        need(len(budgets) == 1 and phys_f == rec["physical_focus_calls"], "Focus-2 visual budgets differed")

        phys_q = 0
        for arm in ARMS:
            base_obs = prior_case["cells"][arm]["native"]["observation"]
            base_ans = (prior_case["cells"][arm]["native"].get("qa") or {}).get("answer")
            for mode in MODES:
                cell = rec["cells"][arm]["modes"][mode]
                frow = rec["focus"][mode][arm]
                expected_obs = combine_observation(base_obs, frow["observation"], frow["capped"])
                need(cell["observation"] == expected_obs, "Combined Focus-2 observation changed")
                if cell["qa_reused_from"] is None:
                    verified = _verify_qa_receipt(work, mode, arm, row, reports, expected_obs,
                                                  native, runtime, io, proc.tokenizer, cell["qa"]["record"])
                    need(verified == cell["qa"], "Focus-2 QA receipt mismatch")
                    phys_q += 1
                expected_eq = cell["qa"]["answer"]
                expected_op = expected_eq if plan["arms"][arm]["kind"] == "on_policy" else base_ans
                need(cell["equal_budget_answer"] == expected_eq and cell["on_policy_answer"] == expected_op,
                     "Focus-2 endpoint answer mismatch")
        need(phys_q == rec["unique_qa_calls"], "Focus-2 QA call count mismatch")
        totals.update(cases=1, triggered=1, focus_calls=phys_f, unique_qa=phys_q)

    need(totals["cases"] == len(order) and totals["triggered"] == manifest["triggered_cases"]
         and totals["focus_calls"] == terminal["counts"]["focus_calls"]
         and totals["unique_qa"] == terminal["counts"]["unique_qa"]
         and not s.torch.cuda.is_initialized(), "Incomplete Focus-2 CPU audit")
    record = dict(
        status="zoom_verify_development_audited", terminal=digest(tpath), parent=PRIOR_RELEASE,
        gpu_released=True, peak_bytes=terminal["peak_bytes"], labels=0, holdout=0, training=0, **totals,
    )
    emit(release_sha256=write(ROOT / "release.json", record), **record)


if __name__ == "__main__":
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "preflight" and not args:
        preflight()
    elif mode == "run" and len(args) == 2:
        run(*args)
    elif mode == "release" and not args:
        audit()
    else:
        raise SystemExit("preflight | run PIN UUID | release")
