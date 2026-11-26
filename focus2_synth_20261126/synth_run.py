"""Option-blind Focus-2 chronological visual synthesis + QA-conversion benchmark on 51 dev videos.

Reuses the exact audited 16 Focus-2 frames from ``zoom_verify_20261125`` (zero change to
frame selection, visual budgets, Source, Planner, Receiver, or Focus-1) to isolate the
Focus-to-QA conversion bottleneck diagnosed on both dev and confirm96:
1. ``synth_qa_native``: option-blind cross-frame visual synthesis Focus-2 -> 100% unmodified pure-text QA.
2. ``synth_qa_priority``: same Focus-2 synthesis -> pure-text QA with neutral close-up priority note.
3. ``direct_visual_qa`` (diagnostic ceiling): same 16 Focus-2 frames + Source reports -> direct VLM choice.
"""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
import json
from math import prod
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

from entropy_routing_20261125 import multi_action, zoom_verify_run as zr
from missing_fact_routing_20261124.live_run import digest, emit, load_history, need, write
from missing_fact_routing_20261124.native_case import _historical_receipt
from missing_fact_routing_20261124.native_runtime import QA_TEMPLATE, _receipt
from missing_fact_routing_20261124.score import ANNOTATION_PIN, ARMS, paired


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
PRIOR_ENTROPY = Path("/data1/llj/lljmas/qwen35_cgbench_entropy_routing_20261125_01")
PRIOR_ZOOM = Path("/data1/llj/lljmas/qwen35_cgbench_zoom_verify_20261125_01")
PRIOR_ZOOM_RELEASE = "bfe43947501596982140cae6106945e2430f928b06e0e2ae6ab03efa5e374329"
ANNOTATION = Path("/data1/llj/development/lljvideo/CLAUDEvidoe/CG-Bench/cgbench.json")
SYNTH_CAP = 512
QA_CAP = 16
NO_INSUFFICIENT = "insufficient visual evidence in this clip"

SYNTH_INSTRUCTION = (
    "Inspect all frames above in chronological order and answer the question based ONLY on "
    "what is directly visible in this clip (including exact counts, colors, on-screen text or "
    "subtitles, objects, and sequence of actions). "
    "Do NOT repeat the question and do NOT output repetitive per-frame lines. "
    "First briefly describe the concrete visual evidence you observe across the clip (citing key "
    "timestamps like 12.3s where helpful), then end with a single final line:\n"
    "Direct visual answer: <concise factual answer, or 'Insufficient visual evidence in this clip'>"
)

PRIORITY_NOTE = (
    "\n\nNote on evidence priority: Observation 3 contains a targeted high-frame-rate close-up "
    "inspection of the specific scene. If Observation 3 reports concrete, directly observed visual "
    "facts (such as exact counts, colors, text/subtitles, or actions) that conflict with a vague "
    "or uncertain overview in Observations 1–2, prioritize Observation 3."
)


def read(path, pin=None):
    if pin is not None:
        need(digest(path) == pin, f"Changed receipt: {Path(path).name}")
    return json.loads(Path(path).read_bytes())


def synth_focus_messages(clocks, question):
    """Build option-blind Focus-2 visual synthesis messages over exact timestamped frames."""
    content = [{"type": "text", "text": "Sampled video frames (timestamps below):\n"}]
    for clk in clocks:
        content.append({"type": "text", "text": clk})
        content.append({"type": "image"})
    content.append({"type": "text", "text": f"Question:\n{question}\n\n{SYNTH_INSTRUCTION}"})
    return [{"role": "user", "content": content}]


def direct_visual_qa_messages(clocks, case, reports, f1_obs):
    """Diagnostic ceiling: direct multiple-choice VLM answer using exact same 16 Focus-2 frames + text."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    opts = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(case["choices"]))
    content = [{"type": "text", "text": "Targeted close-up video frames (timestamps below):\n"}]
    for clk in clocks:
        content.append({"type": "text", "text": clk})
        content.append({"type": "image"})
    body = (
        f"Overview report 1:\n{reports[0]}\n\n"
        f"Overview report 2:\n{reports[1]}\n\n"
        f"Initial clip notes:\n{f1_obs or 'None'}\n\n"
        f"Question:\n{case['question']}\n\n"
        f"Options:\n{opts}\n\n"
        "Answer using the close-up frames above (prioritizing directly visible facts in the frames "
        "over overview guesses). Reply with ONLY the single option letter (e.g., A)."
    )
    content.append({"type": "text", "text": body})
    return [{"role": "user", "content": content}]


def priority_qa_messages(native, io, case, reports, observation):
    """Pure-text QA with one neutral close-up evidence priority sentence (0 pixels)."""
    msgs = io.qa_messages(native, case, reports, observation)
    out = [dict(msgs[0]), dict(msgs[1])]
    out[0]["content"] = out[0]["content"] + PRIORITY_NOTE
    return out


def combine_synth_observation(f1_obs, synth_obs, capped):
    """Combine Focus-1 notes with chronological Focus-2 visual synthesis."""
    clean_f1 = (f1_obs or "").strip()
    clean_s = (synth_obs or "").strip()
    if capped or not clean_s:
        return f1_obs
    if not clean_f1 or clean_f1 == zr.NO_CONTENT:
        return "Focused dense inspection:\n" + clean_s
    return clean_f1 + "\n\nFocused dense inspection:\n" + clean_s


def _load_sealed_f2_images(zoom_case_dir, frow, dest_dir):
    """Copy/verify the exact 16 sealed Focus-2 JPEGs from zoom_verify_20261125 into dest_dir/images."""
    src_arm = frow.get("reused_from") or None
    # Locate the exact native_query focus directory that owns the JPEGs
    owner_dir = zoom_case_dir / "focus" / "native_query" / (src_arm if src_arm else frow["_arm"])
    img_dir = dest_dir / "images"
    img_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    records = []
    from PIL import Image
    for h in frow["jpeg_sha256"]:
        src_jpg = owner_dir / "images" / f"{h}.jpg"
        raw = src_jpg.read_bytes()
        need(digest(src_jpg) == h, f"Sealed Focus-2 JPEG digest mismatch: {h}")
        dst_jpg = img_dir / f"{h}.jpg"
        if not dst_jpg.exists():
            dst_jpg.write_bytes(raw)
        with Image.open(dst_jpg) as im:
            records.append(dict(path=f"images/{h}.jpg", sha256=h, size=list(im.size)))
    return records


def _run_qa(work, subdir, arm, messages, choices, native, runtime, historical, model, occupancy, sink):
    qdir = work / subdir / arm
    qdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker = runtime.attach(native, historical, qdir, model, "qa", 1)
    occupancy()
    with redirect_stdout(sink):
        _, ref = worker.generate(f"qa_{arm}", "qa", messages, runtime.COMMON["qa_cap"])
    occupancy()
    rec = _receipt(qdir, ref, "qa", f"qa_{arm}", messages, runtime.COMMON["qa_cap"])
    tpl = model.tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(not rec["images"] and rec["rendered"] == tpl, "Pure-text QA mismatch")
    ans = None if rec["generation_cap"] else native.option(rec["text"], choices)
    return dict(answer=ans, capped=rec["generation_cap"], record=ref,
                input_tokens=rec["input_tokens"], output_tokens=rec["output_tokens"])


def run_all(uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose one exclusive A40")
    old, d, order = load_history()
    import run_itt_utility as native
    _, _, runtime, io = old.bridge.attach_code()

    zrel = read(PRIOR_ZOOM / "release.json", PRIOR_ZOOM_RELEASE)
    zterm = read(PRIOR_ZOOM / "action" / "terminal.json", zrel["terminal"])
    eterm = read(PRIOR_ENTROPY / "action" / "terminal.json")
    need(zrel["status"] == "zoom_verify_development_audited" and zterm["order"] == order, "Changed parent zoom run")

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
                 and peak <= old.CAP, "GPU occupancy or memory cap violated")

        def stop(*_):
            raise InterruptedError("Focus-2 synthesis run stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700, exist_ok=False)
        totals, committed = Counter(), {}
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP)

        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                versions = tuple(p._version for p in model.model.parameters())

                for i, row in enumerate(d.rows):
                    cid = row.cid
                    work = ACTION / f"case_{i:02d}"
                    work.mkdir(mode=0o700)
                    zdir = PRIOR_ZOOM / "action" / f"case_{i:02d}"
                    edir = PRIOR_ENTROPY / "action" / f"case_{i:02d}"
                    zcase = read(zdir / "case.json", zterm["rows"][cid])
                    ecase = read(edir / "case.json", eterm["rows"][cid])
                    plan = zcase["plan"]
                    reports = [
                        read(edir / k / "outputs" / f"{k}.json")["text"]
                        for k in ("source_shared", "source_query")
                    ]
                    hist = _historical_receipt(row.job["block"])
                    arms_order = multi_action.order_for(i)

                    if not plan["triggered"]:
                        cells = {}
                        for arm in ARMS:
                            base_ans = zcase["cells"][arm]["baseline_answer"]
                            cells[arm] = dict(
                                kind=plan["arms"][arm]["kind"],
                                baseline_f1=base_ans,
                                baseline_f2_op=base_ans,
                                baseline_f2_eq=base_ans,
                                synth_qa_native_op=base_ans,
                                synth_qa_native_eq=base_ans,
                                synth_qa_priority_op=base_ans,
                                synth_qa_priority_eq=base_ans,
                                direct_visual_qa_op=base_ans,
                                direct_visual_qa_eq=base_ans,
                                synth_text=None,
                            )
                        rec = dict(cid=cid, ordinal=i, triggered=False, cells=cells, synth_calls=0, vis_qa_calls=0, qa_calls=0)
                        committed[cid] = write(work / "case.json", rec)
                        continue

                    t0 = perf_counter()
                    seen_synth = {}
                    synth_rows = {}
                    vis_qa_rows = {}
                    synth_calls = 0
                    vis_qa_calls = 0
                    f2_budget = None

                    for arm in arms_order:
                        frow = dict(zcase["focus"]["native_query"][arm], _arm=arm)
                        vkey = tuple(frow["indices"])
                        if vkey in seen_synth:
                            src_arm = seen_synth[vkey]
                            synth_rows[arm] = dict(synth_rows[src_arm], reused_from=src_arm)
                        else:
                            sdir = work / "synth_f2" / arm
                            sdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                            worker = runtime.attach(native, hist, sdir, model, "focus", 1)
                            img_recs = _load_sealed_f2_images(zdir, frow, sdir)
                            msgs = synth_focus_messages(frow["clocks"], row.case["question"])
                            occupancy()
                            with redirect_stdout(sink):
                                srec, sref = worker.generate(f"focus_{arm}", "focus", msgs, SYNTH_CAP, img_recs)
                            occupancy()
                            proof = _receipt(sdir, sref, "focus", f"focus_{arm}", msgs, SYNTH_CAP)
                            bud = (len(img_recs), sum(map(prod, proof["image_grid_thw"])), sum(proof["mm_token_type_ids"]))
                            if f2_budget is None:
                                f2_budget = bud
                            need(bud == f2_budget, "Synthesis Focus-2 visual budget differed")
                            synth_rows[arm] = dict(text=proof["text"], capped=proof["generation_cap"], record=sref, reused_from=None)
                            seen_synth[vkey] = arm
                            synth_calls += 1

                        # Direct visual QA ceiling (keyed by (vkey, base_obs))
                        base_obs = zcase["cells"][arm]["baseline_observation"]
                        vqa_key = (vkey, base_obs)
                        if vqa_key in vis_qa_rows:
                            vis_qa_rows[arm] = vis_qa_rows[vqa_key]
                        else:
                            vdir = work / "direct_vqa" / arm
                            vdir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                            worker = runtime.attach(native, hist, vdir, model, "focus", 1)
                            img_recs = _load_sealed_f2_images(zdir, frow, vdir)
                            vmsgs = direct_visual_qa_messages(frow["clocks"], row.case, reports, base_obs)
                            occupancy()
                            with redirect_stdout(sink):
                                _, vref = worker.generate(f"focus_{arm}", "focus", vmsgs, QA_CAP, img_recs)
                            occupancy()
                            vproof = _receipt(vdir, vref, "focus", f"focus_{arm}", vmsgs, QA_CAP)
                            vans = None if vproof["generation_cap"] else native.option(vproof["text"], row.case["choices"])
                            vout = dict(answer=vans, text=vproof["text"], capped=vproof["generation_cap"], record=vref)
                            vis_qa_rows[vqa_key] = vout
                            vis_qa_rows[arm] = vout
                            vis_qa_calls += 1

                    # Pure-text QA (native & priority, exact-input deduplicated)
                    qa_cache = {}
                    qa_calls = 0
                    cells = {}
                    for arm in arms_order:
                        kind = plan["arms"][arm]["kind"]
                        base_ans = zcase["cells"][arm]["baseline_answer"]
                        base_obs = zcase["cells"][arm]["baseline_observation"]
                        base_f2 = zcase["cells"][arm]["modes"]["native_query"]
                        srow = synth_rows[arm]
                        comb = combine_synth_observation(base_obs, srow["text"], srow["capped"])

                        # 1) Unmodified native pure-text QA
                        n_msgs = io.qa_messages(native, row.case, reports, comb)
                        n_key = json.dumps(n_msgs, sort_keys=True, ensure_ascii=False)
                        if n_key not in qa_cache:
                            qa_cache[n_key] = _run_qa(work, "qa_native", arm, n_msgs, row.case["choices"],
                                                      native, runtime, hist, model, occupancy, sink)
                            qa_calls += 1
                        n_eq = qa_cache[n_key]["answer"]
                        n_op = n_eq if kind == "on_policy" else base_ans

                        # 2) Close-up priority pure-text QA
                        p_msgs = priority_qa_messages(native, io, row.case, reports, comb)
                        p_key = json.dumps(p_msgs, sort_keys=True, ensure_ascii=False)
                        if p_key not in qa_cache:
                            qa_cache[p_key] = _run_qa(work, "qa_priority", arm, p_msgs, row.case["choices"],
                                                      native, runtime, hist, model, occupancy, sink)
                            qa_calls += 1
                        p_eq = qa_cache[p_key]["answer"]
                        p_op = p_eq if kind == "on_policy" else base_ans

                        # 3) Direct visual QA diagnostic
                        v_eq = vis_qa_rows[arm]["answer"]
                        v_op = v_eq if kind == "on_policy" else base_ans

                        cells[arm] = dict(
                            kind=kind,
                            baseline_f1=base_ans,
                            baseline_f2_op=base_f2["on_policy_answer"],
                            baseline_f2_eq=base_f2["equal_budget_answer"],
                            synth_qa_native_op=n_op,
                            synth_qa_native_eq=n_eq,
                            synth_qa_priority_op=p_op,
                            synth_qa_priority_eq=p_eq,
                            direct_visual_qa_op=v_op,
                            direct_visual_qa_eq=v_eq,
                            synth_text=srow["text"],
                            synth_capped=srow["capped"],
                        )

                    rec = dict(
                        cid=cid, ordinal=i, triggered=True, cells=cells,
                        synth_calls=synth_calls, vis_qa_calls=vis_qa_calls, qa_calls=qa_calls,
                        elapsed_seconds=round(perf_counter() - t0, 2),
                    )
                    committed[cid] = write(work / "case.json", rec)
                    totals.update(triggered=1, synth_calls=synth_calls, vis_qa_calls=vis_qa_calls, qa_calls=qa_calls)
                    emit(status="case_committed", ordinal=i, cid=cid, done=len(committed),
                         elapsed_s=rec["elapsed_seconds"], peak_bytes=peak, **totals)

                need(tuple(p._version for p in model.model.parameters()) == versions and len(committed) == 51,
                     "Model weights or cohort changed")
                occupancy()
        except BaseException as err:
            write(ACTION / "terminal.json", dict(status="stopped", error=type(err).__name__, cases=len(committed),
                                                 rows=committed, counts=totals, peak_bytes=peak, pid=os.getpid(), gpu=uuid))
            raise

        write(ACTION / "terminal.json", dict(status="complete", order=order, cases=len(committed), rows=committed,
                                             counts=totals, peak_bytes=peak, pid=os.getpid(), gpu=uuid,
                                             parent=PRIOR_ZOOM_RELEASE, holdout=0, training=0))
        emit(status="inference_complete", cases=len(committed), peak_bytes=peak, **totals)


def score_dev():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Score on CPU with hidden GPUs")
    old, d, order = load_history()
    import run_itt_utility as native
    term = read(ACTION / "terminal.json")
    need(term["status"] == "complete" and term["cases"] == 51, "Run incomplete")
    pid = str(term["pid"])
    proc = Path("/proc") / pid / "stat"
    need((not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z")
         and pid not in old.smi(term["gpu"], "--query-compute-apps=pid").split(), "GPU still running")

    # Decode dev gold answers (exposed 51 dev cohort only; confirmation untouched)
    ns = {}
    exec(compile((PRIOR_ENTROPY / "scoring/metadata_projection_scanner.py").read_bytes(), "proj", "exec"), ns)
    raw_ann = ANNOTATION.read_bytes()
    need(digest(ANNOTATION) == ANNOTATION_PIN, "Annotation changed")
    scanner = ns["Projection"](raw_ann.decode("utf-8"))
    by_uid = {r.case["video_uid"]: r for r in d.rows}
    gold = {}
    for spans in scanner.records():
        uid = scanner.value(spans, "video_uid", {"video_uid"})
        if uid not in by_uid:
            continue
        row = by_uid[uid]
        if scanner.value(spans, "question", {"question"}) != row.case["question"]:
            continue
        ans = scanner.value(spans, "right_answer", {"right_answer"})
        need(native.option(ans, row.case["choices"]) == ans, "Invalid gold")
        gold[row.cid] = ans
    need(len(gold) == 51, "Incomplete dev gold")

    ENDPOINTS = (
        "baseline_f1", "baseline_f2_op", "baseline_f2_eq",
        "synth_qa_native_op", "synth_qa_native_eq",
        "synth_qa_priority_op", "synth_qa_priority_eq",
        "direct_visual_qa_op", "direct_visual_qa_eq",
    )
    items = []
    case_diffs = []
    for i, row in enumerate(d.rows):
        rec = read(ACTION / f"case_{i:02d}" / "case.json", term["rows"][row.cid])
        g = gold[row.cid]
        ans = {}
        for arm in ARMS:
            cell = rec["cells"][arm]
            for ep in ENDPOINTS:
                ans[arm, ep] = (cell[ep] == g)
        items.append((ans, None))
        if rec["triggered"]:
            cs = rec["cells"]["state_split"]
            ct = rec["cells"]["text"]
            cc = rec["cells"]["clock_split"]
            if (cs["synth_qa_native_op"] == g) != (cs["baseline_f2_op"] == g) or (cs["direct_visual_qa_op"] == g) != (ct["direct_visual_qa_op"] == g):
                case_diffs.append(dict(
                    ordinal=i, cid=row.cid, gold=g,
                    base_op=(cs["baseline_f2_op"], ct["baseline_f2_op"], cc["baseline_f2_op"]),
                    synth_nat_op=(cs["synth_qa_native_op"], ct["synth_qa_native_op"], cc["synth_qa_native_op"]),
                    synth_pri_op=(cs["synth_qa_priority_op"], ct["synth_qa_priority_op"], cc["synth_qa_priority_op"]),
                    direct_vis_op=(cs["direct_visual_qa_op"], ct["direct_visual_qa_op"], cc["direct_visual_qa_op"]),
                    synth_snip=(cs.get("synth_text") or "")[:220],
                ))

    out = dict(
        status="focus2_synth_dev_scored",
        split="development",
        n=51,
        holdout=0,
        gpu_released=True,
        peak_bytes=term["peak_bytes"],
        counts=term["counts"],
        endpoints={
            ep: dict(
                correct={arm: sum(a[arm, ep] for a, _ in items) for arm in ARMS},
                state_vs_text=paired(items, ("state_split", ep), ("text", ep)),
                state_vs_clock=paired(items, ("state_split", ep), ("clock_split", ep)),
                clock_vs_text=paired(items, ("clock_split", ep), ("text", ep)),
            )
            for ep in ENDPOINTS
        },
        notable_cases=case_diffs,
    )
    pin = write(ROOT / "score.json", out)
    print(json.dumps(dict(score_sha256=pin, **out), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "run" and len(args) == 1:
        run_all(args[0])
    elif mode == "score" and not args:
        score_dev()
    else:
        raise SystemExit("run UUID | score")
