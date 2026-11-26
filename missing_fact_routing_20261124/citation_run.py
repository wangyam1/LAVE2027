"""Verify cached Focus claims on one A40, then audit the same-frame intervention."""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

from . import citation_check, citation_inputs
from .live_run import digest, emit, load_history, need, write
from .native_case import _historical_receipt
from .native_runtime import QA_TEMPLATE, _receipt
from .report import UNRESOLVED_REPORT


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PRIOR = Path("/data1/llj/lljmas/qwen35_cgbench_missing_fact_20261124_01")
PRIOR_RELEASE = "fdc459c25af96fae4d1a0b4f240f9b2c7c9212871605deb5914098a573bed33a"
PRIOR_SCORE = "f695cb4f443615c182ea037f7e9be924f9574770e02f4b2078553894690af28e"
PREFLIGHT = ROOT / "preflight.json"
ACTION = ROOT / "action"
ARMS = ("text", "state_split", "clock_split")
VERDICTS = {"SUPPORT", "REJECT", "UNCERTAIN", "INVALID"}
FILES = (
    "citation_run.py", "citation_inputs.py", "citation_check.py", "live_run.py", "native_case.py",
    "native_runtime.py", "audit_run.py", "experiment.py", "report.py", "target.py", "policy.py", "receiver.py", "foil_plan.py",
)


def files():
    return {name: digest(HERE / name) for name in FILES}


def read(path, pin=None):
    if pin is not None:
        need(digest(path) == pin, f"Edited receipt: {path.name}")
    return json.loads(path.read_bytes())


def inputs():
    old, d, order = load_history()
    import run_itt_utility as native

    _, _, runtime, io = old.bridge.attach_code()
    scope = citation_inputs.load_citation_inputs(PRIOR, PRIOR_RELEASE, rows=d.rows, native=native, io=io)
    need([z["cid"] for z in scope] == order, "Cached claim roster changed")
    for case in scope:
        case["plans"] = {claim["origin"]: citation_check.prepare_claim(claim) for claim in case["candidates"]}
    release = read(PRIOR / "release.json", PRIOR_RELEASE)
    terminal = read(PRIOR / "action" / "terminal.json", release["terminal"])
    return old, d, order, native, runtime, io, scope, terminal


def roster_of(scope):
    return {
        row["cid"]: {
            origin: sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()
            for origin, plan in row["plans"].items()
        }
        for row in scope
    }


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA during preflight")
    _, d, order, _, _, _, scope, terminal = inputs()
    need(not d.s.torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    # Same report/arm/order for all 51 rows, with the two already-exposed
    # engineering cases first to exercise a supported and an unsupported claim.
    warmup = (16, 4)
    sequence = [*warmup, *(i for i in range(len(order)) if i not in warmup)]
    roster = roster_of(scope)
    record = dict(status="citation_inputs_frozen", files=files(), order=order, sequence=sequence,
                  parent=PRIOR_RELEASE, parent_rows=terminal["rows"], roster=roster, labels=0, holdout=0, gpu=0)
    pin = write(PREFLIGHT, record)
    emit(status=record["status"], cases=len(order), claims=sum(map(len, roster.values())), pin=pin)


def previous(row, ordinal, terminal):
    return read(PRIOR / "action" / f"case_{ordinal:02d}" / "case.json", terminal["rows"][row.cid])


def summary(candidate, result):
    d = result["decisions"]
    need(set(d) == {"cited", "decoy"} and d["cited"] in VERDICTS
         and (d["decoy"] is None or d["decoy"] in VERDICTS), "Unknown verifier decisions")
    expected = candidate["observation"] if d["cited"] == "SUPPORT" else UNRESOLVED_REPORT
    need(result["observation"] == expected, "Verification changed the original report")
    return dict(cited_decision=d["cited"], decoy_decision=d["decoy"], measurement=result)


def gate_cells(prior, scope, claims, fallback):
    original = {arm: prior["cells"][arm]["targeted"] for arm in ARMS}
    cells = {
        arm: dict(original_answer=(value["qa"] or {}).get("answer"),
                  answer=(value["qa"] or {}).get("answer"),
                  gate="bypass", cited_decision=None, decoy_decision=None)
        for arm, value in original.items()
    }
    affected = set()
    for candidate in scope["candidates"]:
        note = claims[candidate["origin"]]
        cited, decoy = note["cited_decision"], note["decoy_decision"]
        for arm in candidate["arms"]:
            need(arm not in affected and original[arm]["status"] == "observed"
                 and original[arm]["observation"] == candidate["observation"], "Citation roster changed")
            affected.add(arm)
            cells[arm].update(gate="retained" if cited == "SUPPORT" else "rejected",
                              cited_decision=cited, decoy_decision=decoy)
    need(affected == {arm for arm, cell in original.items() if cell["status"] == "observed"},
         "Missing cited observation")
    if any(cell["gate"] == "rejected" for cell in cells.values()):
        fallback_qa = fallback()
        for cell in cells.values():
            if cell["gate"] == "rejected":
                cell["answer"] = fallback_qa["answer"]
    else:
        fallback_qa = None
    return cells, fallback_qa


def reusable_fallback(prior):
    for mode in ("targeted", "native"):
        for arm in ARMS:
            same = prior["cells"][arm][mode]
            if same["observation"] == UNRESOLVED_REPORT and same["qa"] is not None:
                return dict(kind="reused_prior", mode=mode, arm=arm, answer=same["qa"]["answer"])
    return None


def fallback_receipt(work, runtime, io, native, row, reports, tokenizer, ref):
    messages = io.qa_messages(native, row.case, reports, UNRESOLVED_REPORT)
    key = "qa_citation_rejected"
    proof = _receipt(work / "fallback_qa", ref, "qa", key, messages, runtime.COMMON["qa_cap"])
    text = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(proof["rendered"] == text and proof["input_ids"] == tokenizer.encode(text, add_special_tokens=False)
         and not proof["images"], "Fallback QA changed its pure-text input")
    answer = None if proof["generation_cap"] else native.option(proof["text"], row.case["choices"])
    return dict(kind="generated", answer=answer, capped=proof["generation_cap"], record=ref)


def run(pin, uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose exactly one A40")
    manifest = read(PREFLIGHT, pin)
    need(manifest["files"] == files() and manifest["labels"] == manifest["holdout"] == 0, "Changed frozen code")
    old, d, order, native, runtime, io, scope, parent_terminal = inputs()
    need(manifest["order"] == order and manifest["parent_rows"] == parent_terminal["rows"]
         and manifest["roster"] == roster_of(scope), "Changed prior scope or decoy plan")
    t = d.s.torch
    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, utilization = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        need(name == "NVIDIA A40" and int(free) >= 36864 and int(utilization) <= 15
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
                 and peak <= old.CAP, "GPU ownership or cap changed")

        def stop(*_):
            raise InterruptedError("Citation verification stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700)
        completed, totals = {}, Counter()
        terminal = dict(order=order, pin=pin, parent=PRIOR_RELEASE, cards=1, gpu=uuid, pid=os.getpid(),
                        labels=0, holdout=0, training=0)
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                versions = tuple(p._version for p in model.model.parameters())
                for i in manifest["sequence"]:
                    row, plan = d.rows[i], scope[i]
                    start = perf_counter()
                    work = ACTION / f"case_{i:02d}"
                    work.mkdir(mode=0o700)
                    emit(status="case_start", ordinal=i, cases=len(completed), candidates=len(plan["candidates"]))
                    history = _historical_receipt(row.job["block"])
                    prior = previous(row, i, parent_terminal)
                    claims = {}
                    for claim in plan["candidates"]:
                        origin = claim["origin"]
                        result = citation_check.verify_claim(
                            plan["plans"][origin], work=work / "claims" / origin, runtime=runtime, native=native,
                            historical=history, model=model, occupancy=occupancy, sink=sink,
                        )
                        claims[origin] = summary(claim, result)

                    def fallback():
                        reused = reusable_fallback(prior)
                        if reused is not None:
                            return reused
                        messages = io.qa_messages(native, row.case, plan["reports"], UNRESOLVED_REPORT)
                        worker = runtime.attach(native, history, work / "fallback_qa", model, "qa", 1)
                        occupancy()
                        with redirect_stdout(sink):
                            _, ref = worker.generate("qa_citation_rejected", "qa", messages, runtime.COMMON["qa_cap"])
                        occupancy()
                        return fallback_receipt(work, runtime, io, native, row, plan["reports"], model.tokenizer, ref)

                    cells, fallback_qa = gate_cells(prior, plan, claims, fallback)
                    values = list(claims.values())
                    totals.update(claims=len(values),
                                  cited_support=sum(v["cited_decision"] == "SUPPORT" for v in values),
                                  decoy_support=sum(v["decoy_decision"] == "SUPPORT" for v in values),
                                  decoy_available=sum(v["decoy_decision"] is not None for v in values),
                                  fallback_qa=int(fallback_qa is not None and fallback_qa["kind"] == "generated"))
                    completed[row.cid] = write(work / "case.json", dict(
                        cid=row.cid, ordinal=i, case_uid=row.case["video_uid"], claims=claims,
                        cells=cells, fallback_qa=fallback_qa, seconds=perf_counter() - start,
                    ))
                    emit(status="case_committed", ordinal=i, cases=len(completed), peak_bytes=peak, **totals)
                need(tuple(p._version for p in model.model.parameters()) == versions, "Model weights changed")
                occupancy()
        except BaseException as error:
            write(ACTION / "terminal.json", dict(status="stopped", error=type(error).__name__, cases=len(completed),
                                                rows=completed, counts=totals, peak_bytes=peak, **terminal))
            raise
        write(ACTION / "terminal.json", dict(status="complete", cases=len(completed), rows=completed,
                                            counts=totals, peak_bytes=peak, **terminal))
        emit(status="inference_complete", cases=len(completed), peak_bytes=peak, **totals)


def audit(prefix=False):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide CUDA for the audit")
    manifest = read(PREFLIGHT)
    old, d, order, native, runtime, io, scope, parent_terminal = inputs()
    need(manifest["files"] == files() and manifest["order"] == order and manifest["roster"] == roster_of(scope)
         and manifest["parent_rows"] == parent_terminal["rows"], "Changed intervention bindings")
    terminal = None if prefix else read(ACTION / "terminal.json")
    if terminal is not None:
        pid = str(terminal["pid"])
        stat = Path("/proc") / pid / "stat"
        need(terminal["status"] == "complete" and terminal["cases"] == len(order) == len(terminal["rows"])
             and terminal["order"] == order and terminal["pin"] == digest(PREFLIGHT)
             and terminal["labels"] == terminal["holdout"] == 0
             and (not stat.exists() or stat.read_text().rsplit(")", 1)[1].split()[0] == "Z")
             and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(),
             "Await the complete single-card job and release")
    s = d.s
    s.torch.set_num_threads(2)
    proc = s.prepare_native(s.AutoProcessor.from_pretrained(s.old.n.MODEL, local_files_only=True), d.nat)
    counts = Counter()
    for i, (row, plan) in enumerate(zip(d.rows, scope, strict=True)):
        work = ACTION / f"case_{i:02d}"
        path = work / "case.json"
        if prefix and not path.is_file():
            continue
        record = read(path, terminal["rows"][row.cid] if terminal is not None else None)
        need(record["cid"] == row.cid and record["ordinal"] == i and record["case_uid"] == row.case["video_uid"],
             "Wrong intervention case")
        prior = previous(row, i, parent_terminal)
        claims = {}
        for claim in plan["candidates"]:
            origin = claim["origin"]
            measurement = citation_check.audit_claim(
                plan["plans"][origin], work=work / "claims" / origin, tokenizer=proc.tokenizer, processor=proc,
            )
            claims[origin] = summary(claim, measurement)
        need(claims == record["claims"], "Citation measurements did not replay")

        def fallback():
            reused = reusable_fallback(prior)
            return reused if reused is not None else fallback_receipt(
                work, runtime, io, native, row, plan["reports"], proc.tokenizer, record["fallback_qa"]["record"],
            )

        cells, fallback_qa = gate_cells(prior, plan, claims, fallback)
        need(cells == record["cells"] and fallback_qa == record["fallback_qa"], "QA intervention changed")
        counts.update(cases=1, claims=len(claims), rejected=sum(c["gate"] == "rejected" for c in cells.values()))
    need(not s.torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    if prefix:
        emit(status="citation_check_prefix_audited", **counts)
        return
    need(counts["cases"] == len(order) and counts["claims"] == terminal["counts"]["claims"],
         "Intervention audit coverage differs")
    result = dict(status="citation_check_development_audited", terminal=digest(ACTION / "terminal.json"),
                  parent=PRIOR_RELEASE, peak_bytes=terminal["peak_bytes"], gpu_released=True,
                  labels=0, holdout=0, training=0, **counts)
    release_pin = write(ROOT / "release.json", result)
    emit(release_sha256=release_pin, **result)


if __name__ == "__main__":
    os.umask(0o077)
    mode, *args = sys.argv[1:]
    if mode == "preflight" and not args:
        preflight()
    elif mode == "run" and len(args) == 2:
        run(*args)
    elif mode in {"release", "prefix"} and not args:
        audit(prefix=mode == "prefix")
    else:
        raise SystemExit("preflight | run PIN UUID | prefix | release")
