"""Reacquire eight new frames after unresolved citation verification."""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys

from . import citation_check, citation_run, refocus_report
from .live_run import digest, emit, need, write
from .native_case import _historical_receipt
from .native_runtime import QA_TEMPLATE, _receipt
from .refocus_sampling import plan_refocus


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
MANIFEST = ROOT / "preflight.json"
CITATION = Path("/data1/llj/lljmas/qwen35_cgbench_citation_check_20261125_02")
CITATION_RELEASE = "731ff8b83014f85bd4ac2140994ce4eaeeab8bf52411f72899e1687b59909556"
ARMS = citation_run.ARMS
SAMPLES = ("local", "clock")
CAP = 160
CODE = (
    "refocus_run.py", "refocus_sampling.py", "refocus_report.py", "citation_check.py", "citation_run.py",
    "citation_inputs.py", "audit_run.py", "live_run.py", "native_case.py", "native_runtime.py", "experiment.py",
    "target.py", "policy.py", "receiver.py", "report.py", "foil_plan.py",
)
read = citation_run.read


def code():
    return {name: digest(HERE / name) for name in CODE}


def seal(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def load():
    old, d, order, native, runtime, io, cases, previous = citation_run.inputs()
    _, triad, _, _ = old.bridge.attach_code()
    loss = read(CITATION / "release.json", CITATION_RELEASE)
    terminal = read(CITATION / "action" / "terminal.json", loss["terminal"])
    need(loss["status"] == "citation_check_development_audited" and loss["gpu_released"] is True
         and terminal["status"] == "complete" and terminal["order"] == order
         and terminal["labels"] == terminal["holdout"] == terminal["training"] == 0, "Changed citation experiment")
    programs = []
    for i, (row, case) in enumerate(zip(d.rows, cases, strict=True)):
        prior = citation_run.previous(row, i, previous)
        cited_case = read(CITATION / "action" / f"case_{i:02d}" / "case.json", terminal["rows"][row.cid])
        need(cited_case["cid"] == case["cid"] == row.cid and cited_case["ordinal"] == i
             and cited_case["case_uid"] == row.case["video_uid"], "Cited case moved")
        choices = {}
        for view in case["candidates"]:
            arm = view["origin"]
            claim = cited_case["claims"][arm]
            if claim["cited_decision"] != "UNCERTAIN":
                continue
            original = prior["routes"]["arguments"][arm]
            shared_frames = sorted({frame for values in prior["routes"]["indices"].values() for frame in values})
            plan = plan_refocus(
                cited_indices=view["cited_indices"], source_indices=row.media["overview_indices"],
                focus_indices=view["frame_indices"], other_focus_indices=shared_frames,
                fps=row.media["fps"], frame_count=row.media["frame_count"], focus_clips=original["clips"],
            )
            choices[arm] = (view, plan)
        programs.append((row, case, prior, choices))
    return old, d, order, native, runtime, io, triad, programs


def roster(programs):
    return {r.cid: {a: p for a, (_, p) in options.items()} for r, _, _, options in programs}


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for input preflight")
    _, d, order, _, _, _, _, programs = load()
    fixed = roster(programs)
    n = sum(p["status"] == "planned" for row in fixed.values() for p in row.values())
    need(not d.s.torch.cuda.is_initialized() and n > 0, "No eligible equal-budget follow-up plan")
    sequence = [4, 16, *(i for i in range(len(order)) if i not in (4, 16))]
    layouts = {k: sorted(v) for k, v in fixed.items()}
    record = dict(status="refocus_inputs_frozen", code=code(), order=order, sequence=sequence,
                  parent=CITATION_RELEASE, roster=seal(fixed), arms=layouts, claims=n, labels=0, holdout=0, gpu=0)
    pin = write(MANIFEST, record)
    emit(status=record["status"], cases=len(order), claims=n, uncertain=sum(map(len, fixed.values())),
         no_plan=sum(map(len, fixed.values()))-n, pin=pin)


def frame_request(prior, indices, fps):
    return dict(query=prior["text_action"]["query"], start_time=indices[0] / fps, end_time=(indices[-1] + 1) / fps)


def records_from_render(directory, view):
    from PIL import Image

    result = []
    for h in view["jpeg_sha256"]:
        name = f"images/{h}.jpg"
        path = directory / name
        need(digest(path) == h and not path.is_symlink(), "Changed native follow-up input JPEG")
        with Image.open(path) as image:
            need(image.format == "JPEG" and min(image.size) > 0, "Expected a valid follow-up frame")
            result.append(dict(path=name, sha256=h, size=list(image.size)))
    return result


def read_outcome(directory, view, target, fps, ref, tokenizer, processor=None):
    messages = refocus_report.render_refocus_prompt(target, view["indices"], fps)
    proof = _receipt(directory, ref, "focus", "read_refocus", messages, CAP)
    images = records_from_render(directory, view)
    template = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    grids = proof["image_grid_thw"]
    tokens = sum(proof["mm_token_type_ids"])
    need(len(images) == len(grids) == 8 and proof["images"] == images and proof["rendered"] == template
         and tokens > 0, "Refocus generation changed the eight-frame input")
    if processor is not None:
        citation_check._processor_replay(processor, template, directory, images, proof)
    report = refocus_report.parse_refocus_report(proof["text"], view["indices"], fps, capped=proof["generation_cap"])
    return dict(ref=ref, view=view, report=report, budget=dict(image_grid_thw=grids, visual_tokens=tokens,
                                                              frames=len(images))), proof


def verifier_plan(directory, read_result, proof, target, fps):
    parsed = read_result["report"]
    plan = citation_check.prepare_claim(dict(
        target=target, claim=parsed["claim"], observation=parsed["observation"],
        citations=parsed["citations"], frame_indices=read_result["view"]["indices"], images=proof["images"],
        image_grid_thw=proof["image_grid_thw"], directory=directory, fps=fps,
    ))
    # The first experiment already measured decoy selectivity. Here the verifier
    # only gates whether newly reported evidence is sufficiently supported.
    plan["views"]["decoy"] = []
    plan["order"] = ["cited"]
    return plan


def paired_order(i, arm):
    return SAMPLES if (i + ARMS.index(arm)) % 2 == 0 else SAMPLES[::-1]


def evaluate(row, ordinal, case, prior, choices, read_sample, verify, answer):
    cells = {}
    for arm in ARMS:
        before = prior["cells"][arm]["targeted"]
        base = (before["qa"] or {}).get("answer")
        cells[arm] = dict(baseline=base, local=base, clock=base, local_verified=base, clock_verified=base, planned=False)
    claims = {}
    for origin in ARMS:
        if origin not in choices:
            continue
        candidate, plan = choices[origin]
        record = dict(arms=candidate["arms"], plan=plan, reads={})
        claims[origin] = record
        if plan["status"] != "planned":
            continue
        fps = row.media["fps"]
        order = paired_order(ordinal, origin)
        for route in order:
            indices = plan[f"{route}_indices"]
            measurement, proof = read_sample(origin, route, indices, candidate["target"])
            merged = refocus_report.merge_refocus_report(candidate["observation"], candidate["claim"],
                                                          measurement["report"])
            verified = None
            if merged["origin"] == "refocus":
                verified = verify(origin, route, measurement, proof, candidate["target"], fps)
            observation = merged["observation"]
            option = answer(origin, route, observation) if merged["origin"] == "refocus" else None
            record["reads"][route] = dict(measurement=measurement, merge=merged, verification=verified, qa=option)
        local, clock = (record["reads"][k]["measurement"] for k in SAMPLES)
        need(local["budget"] == clock["budget"], "Actual added visual-token budget differs")
        for arm in candidate["arms"]:
            cells[arm]["planned"] = True
            for route, result in record["reads"].items():
                if result["qa"] is not None:
                    cells[arm][route] = result["qa"]["answer"]
                    if result["verification"]["decisions"]["cited"] == "SUPPORT":
                        cells[arm][f"{route}_verified"] = result["qa"]["answer"]
    return cells, claims


def qa_result(work, origin, route, row, case, observation, native, runtime, io, tok, ref):
    directory = work / "qa" / origin / route
    template = io.qa_messages(native, row.case, case["reports"], observation)
    record = _receipt(directory, ref, "qa", "qa_refocus", template, runtime.COMMON["qa_cap"])
    raw = tok.apply_chat_template(template, **QA_TEMPLATE)
    need(record["rendered"] == raw and record["input_ids"] == tok.encode(raw, add_special_tokens=False)
         and not record["images"], "Refocus QA changed the pure-text input")
    answer = None if record["generation_cap"] else native.option(record["text"], row.case["choices"])
    return dict(record=ref, answer=answer)


def live_case(resources, item, i, work, model, occupancy, sink):
    old, d, _, native, runtime, io, triad, _ = resources
    row, case, prior, chooses = item
    history = _historical_receipt(row.job["block"])
    geo, _, reference = old.bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
    reader = None

    def read_sample(origin, route, indices, target):
        nonlocal reader
        import decord
        import focus_helpers as helper
        from live_dev import media_path

        if reader is None:
            reader = decord.VideoReader(str(media_path(row)), ctx=decord.cpu(0), num_threads=1)
        directory = work / "reads" / origin / route
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        worker = runtime.attach(native, history, directory, model, "focus", 1)
        query = frame_request(prior, indices, row.media["fps"])
        rendered = triad.render(native, io, helper, reference, directory, query, indices, reader, geo["fps"], geo["count"])
        view = dict(arguments=query, **{k: rendered[k] for k in ("indices", "clocks", "jpeg_sha256", "rgb_sha256")})
        messages = refocus_report.render_refocus_prompt(target, view["indices"], row.media["fps"])
        occupancy()
        with redirect_stdout(sink):
            _, ref = worker.generate("read_refocus", "focus", messages, CAP, records_from_render(directory, view))
        occupancy()
        return read_outcome(directory, view, target, row.media["fps"], ref, model.tokenizer)

    def verify(origin, route, measurement, proof, target, fps):
        directory = work / "reads" / origin / route
        return citation_check.verify_claim(
            verifier_plan(directory, measurement, proof, target, fps), work=work / "verify" / origin / route,
            runtime=runtime, native=native, historical=history, model=model, occupancy=occupancy, sink=sink,
        )

    def answer(origin, route, observation):
        directory = work / "qa" / origin / route
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        worker = runtime.attach(native, history, directory, model, "qa", 1)
        occupancy()
        with redirect_stdout(sink):
            _, ref = worker.generate("qa_refocus", "qa", io.qa_messages(native, row.case, case["reports"], observation),
                                     runtime.COMMON["qa_cap"])
        occupancy()
        return qa_result(work, origin, route, row, case, observation, native, runtime, io, model.tokenizer, ref)

    return evaluate(row, i, case, prior, chooses, read_sample, verify, answer)


def replay_case(resources, item, i, work, stored, processor):
    """Reconstruct the same physical acquisitions and text-only QA from receipts."""
    _, _, _, native, runtime, io, _, _ = resources
    row, case, prior, choices = item

    def read_sample(origin, route, indices, target):
        rec = stored["claims"][origin]["reads"][route]["measurement"]
        need(rec["view"]["indices"] == indices
             and rec["view"]["arguments"] == frame_request(prior, indices, row.media["fps"]),
             "Physical read did not use frozen frame indices and arguments")
        return read_outcome(work / "reads" / origin / route, rec["view"], target, row.media["fps"],
                            rec["ref"], processor.tokenizer, processor)

    def verify(origin, route, measurement, proof, target, fps):
        directory = work / "reads" / origin / route
        return citation_check.audit_claim(
            verifier_plan(directory, measurement, proof, target, fps), work=work / "verify" / origin / route,
            tokenizer=processor.tokenizer, processor=processor,
        )

    def answer(origin, route, observation):
        ref = stored["claims"][origin]["reads"][route]["qa"]["record"]
        return qa_result(work, origin, route, row, case, observation, native, runtime, io, processor.tokenizer, ref)

    cells, claims = evaluate(row, i, case, prior, choices, read_sample, verify, answer)
    need(cells == stored["cells"] and claims == stored["claims"], "Unreproducible refocus decisions")
    return cells, claims


def check_manifest(resources, pin=None):
    data = read(MANIFEST, pin)
    need(data["code"] == code() and data["order"] == resources[2] and data["parent"] == CITATION_RELEASE
         and data["roster"] == seal(roster(resources[-1])) and data["labels"] == data["holdout"] == data["gpu"] == 0,
         "Changed frozen refocus design")
    return data


def run(pin, uuid, share):
    need(share in {"exclusive", "shared"} and os.environ.get("CUDA_VISIBLE_DEVICES") == uuid
         and uuid.startswith("GPU-") and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native",
         "Expose exactly one protected GPU")
    resources = load()
    old, d, order, *_ = resources
    manifest = check_manifest(resources, pin)
    t = d.s.torch
    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, use = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        others = set(old.smi(uuid, "--query-compute-apps=pid").split())
        need(name == "NVIDIA A40" and (int(free) >= 33 * 1024 if share == "shared"
                                      else int(free) >= 36 * 1024 and not others and int(use) <= 15),
             "A40 does not meet the selected resource policy")
        d.s.old.verify_weights()
        need(t.cuda.device_count() == 1 and not t.cuda.memory_reserved() and t.cuda.get_allocator_backend() == "native",
             "Unexpected CUDA context")
        t.cuda.set_per_process_memory_fraction(old.CAP / t.cuda.get_device_properties(0).total_memory)
        peak = 0

        def occupancy():
            nonlocal peak
            peak = max(peak, t.cuda.max_memory_reserved())
            pids = set(old.smi(uuid, "--query-compute-apps=pid").split())
            need(pids <= {str(os.getpid()), *others} and peak <= old.CAP, "New GPU user or allocation beyond cap")
            if share == "shared":
                need(int(old.smi(uuid, "--query-gpu=memory.free")) >= 2048, "Shared GPU margin exhausted")

        def stop(*_):
            raise InterruptedError("Refocus stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700)
        terminal = dict(order=order, pin=pin, cards=1, gpu=uuid, pid=os.getpid(), share=share,
                        labels=0, holdout=0, training=0)
        complete = {}
        totals = Counter()
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP, share=share)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                versions = tuple(p._version for p in model.model.parameters())
                for i in manifest["sequence"]:
                    item = resources[-1][i]
                    row = item[0]
                    work = ACTION / f"case_{i:02d}"
                    work.mkdir(mode=0o700)
                    cells, claims = live_case(resources, item, i, work, model, occupancy, sink)
                    complete[row.cid] = write(work / "case.json", dict(cid=row.cid, ordinal=i,
                                                     case_uid=row.case["video_uid"], cells=cells, claims=claims))
                    reads = [r for c in claims.values() for r in c["reads"].values()]
                    totals.update(physical_reads=len(reads), qa=sum(r["qa"] is not None for r in reads),
                                  verified=sum(r["verification"] is not None and
                                               r["verification"]["decisions"]["cited"] == "SUPPORT" for r in reads))
                    emit(status="case_committed", ordinal=i, cases=len(complete), peak_bytes=peak, **totals)
                need(tuple(p._version for p in model.model.parameters()) == versions, "Model weights changed")
                occupancy()
        except BaseException as error:
            write(ACTION / "terminal.json", dict(status="stopped", error=type(error).__name__, cases=len(complete),
                                                rows=complete, counts=totals, peak_bytes=peak, **terminal))
            raise
        write(ACTION / "terminal.json", dict(status="complete", cases=len(complete), rows=complete, counts=totals,
                                            peak_bytes=peak, **terminal))
        emit(status="inference_complete", cases=len(complete), peak_bytes=peak, **totals)


def audit(prefix=False):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for audit")
    resources = load()
    old, d, order, *_, programs = resources
    manifest = check_manifest(resources)
    terminal = None if prefix else read(ACTION / "terminal.json")
    if terminal is not None:
        pid = str(terminal["pid"])
        stat = Path("/proc") / pid / "stat"
        need(terminal["status"] == "complete" and terminal["cases"] == len(order) == len(terminal["rows"])
             and terminal["order"] == order and terminal["pin"] == digest(MANIFEST)
             and terminal["labels"] == terminal["holdout"] == 0
             and (not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
             and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(), "Await GPU release")
    s = d.s
    s.torch.set_num_threads(2)
    proc = s.prepare_native(s.AutoProcessor.from_pretrained(s.old.n.MODEL, local_files_only=True), d.nat)
    counts = Counter()
    for i, item in enumerate(programs):
        row = item[0]
        work = ACTION / f"case_{i:02d}"
        path = work / "case.json"
        if prefix and not path.is_file():
            continue
        stored = read(path, terminal["rows"][row.cid] if terminal is not None else None)
        need(stored["cid"] == row.cid and stored["ordinal"] == i and stored["case_uid"] == row.case["video_uid"],
             "Moved refocus case")

        _, claims = replay_case(resources, item, i, work, stored, proc)
        counts.update(cases=1, reads=sum(len(r["reads"]) for r in claims.values()))
    need(not s.torch.cuda.is_initialized(), "CPU refocus audit initialized CUDA")
    if prefix:
        emit(status="refocus_prefix_audited", **counts)
    else:
        need(counts["cases"] == len(order) and counts["reads"] == 2 * manifest["claims"], "Missing planned observations")
        record = dict(status="refocus_development_audited", terminal=digest(ACTION / "terminal.json"),
                      gpu_released=True, labels=0, holdout=0, training=0, share=terminal["share"], **counts)
        emit(release_sha256=write(ROOT / "release.json", record), **record)


if __name__ == "__main__":
    os.umask(0o077)
    action, *args = sys.argv[1:]
    if action == "preflight" and not args:
        preflight()
    elif action == "run" and len(args) == 3:
        run(*args)
    elif action in {"release", "prefix"} and not args:
        audit(prefix=action == "prefix")
    else:
        raise SystemExit("preflight | run PIN UUID (shared|exclusive) | prefix | release")
