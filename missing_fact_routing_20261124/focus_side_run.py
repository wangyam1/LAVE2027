"""Compare Focus-state, text, and clock selection on the same two new-frame windows."""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import signal
import sys

from . import citation_run, focus_side_preflight, focus_side_receiver, focus_state_capture, receiver, refocus_run
from .focus_side_policy import approve_side
from .live_run import digest, emit, need, write
from .native_case import _historical_receipt
from .native_runtime import QA_TEMPLATE, _receipt
from .target import target_from_question


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
MANIFEST = ROOT / "preflight.json"
ARMS = citation_run.ARMS
SIDE_TO_REFOCUS = {"before": "local", "after": "clock"}
CODE = ("focus_side_run.py", "focus_side_preflight.py", "focus_side_policy.py",
        "focus_side_receiver.py", "focus_state_capture.py")
read = citation_run.read


def code():
    return refocus_run.code() | {name: digest(HERE / name) for name in CODE}


def setup():
    resources = refocus_run.load()
    menu = focus_side_preflight.build_menu(resources)
    parent = citation_run.PRIOR
    release = read(parent / "release.json", citation_run.PRIOR_RELEASE)
    terminal = read(parent / "action" / "terminal.json", release["terminal"])
    foils = read(parent / "preflight.json", terminal["pin"])["foils"]
    return resources, menu, foils


def layouts(d, proc, rope, raw, reports, candidate, plan, foil):
    grid, = {tuple(g) for g in candidate["image_grid_thw"]}
    common = focus_side_receiver.context(reports, candidate["observation"], plan, candidate["fps"])
    return [
        receiver.render(d.s, proc, rope, grid, scope, raw["query"], common, 20000, rule=focus_side_receiver.RULE)[0]
        for scope in (candidate["target"], foil)
    ]


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for preflight")
    resources, menu, foils = setup()
    _, d, order, *_ = resources
    s = d.s
    s.torch.set_num_threads(2)
    proc = s.prepare_native(s.AutoProcessor.from_pretrained(s.old.n.MODEL, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(s.old.n.MODEL, local_files_only=True))
    n = 0
    for row, case, prior, candidates in resources[-1]:
        for origin, (candidate, _) in candidates.items():
            plan = menu[row.cid][origin]
            if plan["status"] != "planned":
                continue
            real, wrong = layouts(d, proc, rope, prior["text_action"], case["reports"], candidate, plan,
                                  target_from_question(foils[row.cid]["question"]))
            need(real["context_sha256"] == wrong["context_sha256"] and real["target_sha256"] != wrong["target_sha256"]
                 and real["labels"] == wrong["labels"], "Wrong-target control does not match this prompt")
            n += 1
    need(n > 0 and not s.torch.cuda.is_initialized(), "No eligible label-blind temporal choice")
    seal = dict(status="focus_side_inputs_frozen", order=order, parent=refocus_run.CITATION_RELEASE,
                menu=refocus_run.seal(menu), files=code(), eligible=n, labels=0, holdout=0, gpu=0)
    emit(status=seal["status"], cases=len(order), eligible=n, pin=write(MANIFEST, seal))


def matched_manifest(resources, menu, pin=None):
    seal = read(MANIFEST, pin)
    need(seal["files"] == code() and seal["order"] == resources[2] and seal["menu"] == refocus_run.seal(menu)
         and seal["parent"] == refocus_run.CITATION_RELEASE and seal["labels"] == seal["holdout"] == seal["gpu"] == 0,
         "Changed focus-side design")
    return seal


def selected_item(item, plans):
    row, case, prior, candidates = item
    interventions = {}
    for origin, (candidate, _) in candidates.items():
        plan = plans[origin]
        if plan["status"] == "planned":
            interventions[origin] = (candidate, dict(
                plan, local_indices=plan["menu"]["before"]["frame_indices"],
                clock_indices=plan["menu"]["after"]["frame_indices"],
            ))
    return row, case, prior, interventions


def text_pick_path(work, origin):
    return work / "text_plan" / origin


def text_pick(native, runtime, directory, reference, candidate, prior, case, plan, tokenizer):
    fps = candidate["fps"]
    messages = focus_side_receiver.text_side_messages(
        candidate["target"], prior["text_action"]["query"], case["reports"], candidate["observation"], plan, fps,
    )
    receipt = _receipt(directory, reference, "qa", "f2_text", messages, runtime.COMMON["qa_cap"])
    template = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(not receipt["images"] and receipt["rendered"] == template
         and receipt["input_ids"] == tokenizer.encode(template, add_special_tokens=False),
         "Text Planner did not receive the exact common history")
    return focus_side_receiver.decode_text_side(receipt["text"], receipt["generation_cap"], plan)


def live_decision(resources, item, work, origin, foil, model, occupancy, sink):
    _, d, _, native, runtime, _, _, _ = resources
    row, case, prior, candidates = item
    candidate, plan = candidates[origin]
    hist = _historical_receipt(row.job["block"])
    query = prior["text_action"]["query"]
    fps = candidate["fps"]
    messages = focus_side_receiver.text_side_messages(candidate["target"], query, case["reports"],
                                                      candidate["observation"], plan, fps)
    out = text_pick_path(work, origin)
    out.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker = runtime.attach(native, hist, out, model, "qa", 1)
    occupancy()
    with redirect_stdout(sink):
        _, ref = worker.generate("f2_text", "qa", messages, runtime.COMMON["qa_cap"])
    occupancy()
    text_side = text_pick(native, runtime, out, ref, candidate, prior, case, plan, model.tokenizer)

    from source_capture import capture_source

    states, f1 = focus_state_capture.capture_first_focus(
        candidate, work=work / "capture" / origin, native=native, runtime=runtime, historical=hist,
        model=model, capture=capture_source, occupancy=occupancy, sink=sink,
    )
    probe = None
    if states is not None:
        grid, = {tuple(g) for g in candidate["image_grid_thw"]}
        occupancy()
        probe = focus_side_receiver.measure(d, row, model, states, grid, candidate["target"], foil, query, case["reports"],
                                            candidate["observation"], plan, fps)
        occupancy()
    approved = probe is not None and probe["verdict"]["status"] == "approved"
    side = probe["verdict"]["selected_side"] if approved else text_side
    return dict(arms=candidate["arms"], text_side=text_side, state_side=side, clock_side=plan["clock_side"],
                approved=approved, capture=f1, probe=probe, text_receipt=ref)


def make_cells(base, decisions):
    outcomes = {}
    for arm, cell in base.items():
        outcomes[arm] = dict(baseline=cell["baseline"], before=cell["local"], after=cell["clock"],
                             evaluated=cell["planned"], **{"latent": cell["baseline"], "text": cell["baseline"],
                             "clock": cell["baseline"], "latent_verified": cell["baseline"],
                             "text_verified": cell["baseline"], "clock_verified": cell["baseline"]})
    aliases = {}
    for chosen in decisions.values():
        for arm in chosen["arms"]:
            need(base[arm]["planned"] and arm not in aliases, "Side decision did not bind a physical view")
            aliases[arm] = chosen
    need(set(aliases) == {a for a, v in base.items() if v["planned"]}, "Missing side decisions")
    for arm, chosen in aliases.items():
        for method, field in (("latent", "state_side"), ("text", "text_side"), ("clock", "clock_side")):
            source = SIDE_TO_REFOCUS[chosen[field]]
            outcomes[arm][method] = base[arm][source]
            outcomes[arm][f"{method}_verified"] = base[arm][f"{source}_verified"]
    return outcomes


def run(pin, uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose one exclusive A40")
    resources, menu, foils = setup()
    old, d, order, *_ = resources
    seal = matched_manifest(resources, menu, pin)
    t = d.s.torch
    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, use = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        need(name == "NVIDIA A40" and int(free) >= 36 * 1024 and int(use) <= 15
             and not old.smi(uuid, "--query-compute-apps=pid"), "A40 is not exclusive")
        d.s.old.verify_weights()
        need(t.cuda.device_count() == 1 and not t.cuda.memory_reserved() and t.cuda.get_allocator_backend() == "native",
             "Unexpected CUDA context")
        t.cuda.set_per_process_memory_fraction(old.CAP / t.cuda.get_device_properties(0).total_memory)
        peak = 0

        def occupancy():
            nonlocal peak
            peak = max(peak, t.cuda.max_memory_reserved())
            need(set(old.smi(uuid, "--query-compute-apps=pid").split()) <= {str(os.getpid())} and peak <= old.CAP,
                 "GPU ownership or allocation changed")

        def stop(*_):
            raise InterruptedError("Focus-state experiment stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700)
        done, counts = {}, Counter()
        terminal = dict(order=order, pin=pin, cases=len(order), cards=1, pid=os.getpid(), gpu=uuid,
                        labels=0, holdout=0, training=0)
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                versions = tuple(p._version for p in model.model.parameters())
                for i, source in enumerate(resources[-1]):
                    row = source[0]
                    item = selected_item(source, menu[row.cid])
                    work = ACTION / f"case_{i:02d}"
                    work.mkdir(mode=0o700)
                    foil = target_from_question(foils[row.cid]["question"])
                    # All policy decisions are committed before either F2 window is read.
                    decisions = {
                        origin: live_decision(resources, item, work, origin, foil, model, occupancy, sink)
                        for origin in item[-1]
                    }
                    if decisions:
                        write(work / "choices.json", decisions)
                    base, reads = refocus_run.live_case(resources, item, i, work, model, occupancy, sink)
                    done[row.cid] = write(work / "case.json", dict(
                        cid=row.cid, ordinal=i, case_uid=row.case["video_uid"], cells=make_cells(base, decisions),
                        reads=reads, base=base, decisions=decisions,
                    ))
                    counts.update(eligible=len(decisions), approved=sum(v["approved"] for v in decisions.values()),
                                  matched_f1=sum(v["capture"]["matched_generated_tokens"] for v in decisions.values()),
                                  latent_not_text=sum(v["state_side"] != v["text_side"] for v in decisions.values()))
                    emit(status="case_committed", ordinal=i, cases=len(done), peak_bytes=peak, **counts)
                need(tuple(p._version for p in model.model.parameters()) == versions
                     and counts["eligible"] == seal["eligible"], "Model weights or scope changed")
                occupancy()
        except BaseException as error:
            terminal.update(cases=len(done))
            write(ACTION / "terminal.json", dict(status="stopped", errors=type(error).__name__, rows=done,
                                                counts=counts, peak_bytes=peak, **terminal))
            raise
        write(ACTION / "terminal.json", dict(status="complete", rows=done, counts=counts, peak_bytes=peak, **terminal))
        emit(status="inference_complete", cases=len(done), peak_bytes=peak, **counts)


def verify_decision(resources, item, work, origin, proof, proc, rope, foil):
    _, d, _, native, runtime, _, _, _ = resources
    row, case, prior, candidates = item
    candidate, plan = candidates[origin]
    original = read(candidate["directory"] / candidate["focus"]["path"], candidate["focus"]["sha256"])
    capture = proof["capture"]
    current = _receipt(work / "capture" / origin, capture["record"], "focus", "first_focus",
                       original["messages"], runtime.COMMON["source_cap"])
    matched = current["generated_token_ids"] == original["generated_token_ids"]
    need(all(current[k] == original[k] for k in focus_state_capture.BOUND_FIELDS)
         and capture["matched_generated_tokens"] == matched and capture["frame_indices"] == candidate["frame_indices"]
         and capture["states"] == len(candidate["frame_indices"]) and capture["visual_prefills"] == 1
         and capture["frozen_record"] == candidate["focus"],
         "Capture receipt does not identify the original Focus input")
    text_side = text_pick(native, runtime, text_pick_path(work, origin), proof["text_receipt"], candidate,
                          prior, case, plan, proc.tokenizer)
    approved = False
    state = text_side
    probe = proof["probe"]
    if matched:
        actual, wrong = layouts(d, proc, rope, prior["text_action"], case["reports"], candidate, plan, foil)
        verdict = approve_side(plan, **probe["scores"])
        need(probe["actual_prompt"] == actual and probe["wrong_prompt"] == wrong and probe["verdict"] == verdict
             and probe["forwards"] == 10 and probe["visual_forwards"] == 0, "State evidence or controls changed")
        approved = verdict["status"] == "approved"
        state = verdict["selected_side"] if approved else text_side
    else:
        need(probe is None, "Drifted Focus output cannot use a state intervention")
    need(proof["approved"] == approved and proof["clock_side"] == plan["clock_side"]
         and proof["text_side"] == text_side and proof["state_side"] == state and proof["arms"] == candidate["arms"],
         "Physical side selection did not reproduce")


def audit():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for audit")
    resources, menu, foils = setup()
    old, d, order, *_ = resources
    seal = matched_manifest(resources, menu)
    path = ACTION / "terminal.json"
    terminal = read(path)
    pid = str(terminal["pid"])
    stat = Path("/proc") / pid / "stat"
    need(terminal["status"] == "complete" and terminal["order"] == order and terminal["cases"] == len(order)
         and terminal["pin"] == digest(MANIFEST) and terminal["labels"] == terminal["holdout"] == 0
         and (not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
         and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(), "Await GPU release")
    s = d.s
    s.torch.set_num_threads(2)
    proc = s.prepare_native(s.AutoProcessor.from_pretrained(s.old.n.MODEL, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(s.old.n.MODEL, local_files_only=True))
    totals = Counter()
    for i, source in enumerate(resources[-1]):
        row = source[0]
        item = selected_item(source, menu[row.cid])
        work = ACTION / f"case_{i:02d}"
        result = read(work / "case.json", terminal["rows"][row.cid])
        need(result["cid"] == row.cid and result["case_uid"] == row.case["video_uid"] and result["ordinal"] == i
             and set(result["decisions"]) == set(item[-1]), "Case or physical decisions changed")
        if item[-1]:
            need(read(work / "choices.json") == result["decisions"], "Choices were changed after F2")
        for origin, proof in result["decisions"].items():
            verify_decision(resources, item, work, origin, proof, proc, rope, target_from_question(foils[row.cid]["question"]))
        base, reads = refocus_run.replay_case(resources, item, i, work,
                                             dict(cells=result["base"], claims=result["reads"]), proc)
        need(make_cells(base, result["decisions"]) == result["cells"], "Final endpoint changed")
        totals.update(eligible=len(reads), reads=sum(len(v["reads"]) for v in reads.values()))
    need(totals["eligible"] == seal["eligible"] and totals["reads"] == 2 * seal["eligible"]
         and not s.torch.cuda.is_initialized(), "Incomplete CPU audit")
    record = dict(status="focus_side_development_audited", terminal=digest(path), cases=len(order),
                  gpu_released=True, labels=0, holdout=0, training=0, **totals)
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
