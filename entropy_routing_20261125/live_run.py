"""Preflight and execute the entropy-guided multi-interval LAVE development experiment on A40."""

from collections import Counter
from contextlib import redirect_stdout
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from time import perf_counter

from missing_fact_routing_20261124 import receiver
from missing_fact_routing_20261124.foil_plan import plan_foils
from missing_fact_routing_20261124.live_run import digest, emit, load_history, need, write
from missing_fact_routing_20261124.target import target_from_question

from . import multi_action, policy
from .native_case import run_live_case


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ACTION = ROOT / "action"
PREFLIGHT = ROOT / "preflight.json"
CODE = (
    "live_run.py", "audit_run.py", "native_case.py", "native_runtime.py",
    "experiment.py", "multi_action.py", "policy.py", "report.py",
)
SHARED_CODE = ("receiver.py", "target.py", "foil_plan.py", "live_run.py", "native_runtime.py")


def files():
    shared_dir = ROOT / "missing_fact_routing_20261124"
    own = {f"entropy/{name}": digest(HERE / name) for name in CODE}
    shared = {f"shared/{name}": digest(shared_dir / name) for name in SHARED_CODE}
    return own | shared


def engineering_order(rows):
    """Run two motivating exposed cases plus two newly-unlocked/multi-frame cases first, keeping all ordinals."""
    first = []
    # 1. Jinhao 80 price (ordinal 16), 2. Green onions count (ordinal 4)
    for fragments in (("jinhao 80", "price"), ("green onions", "pick up")):
        matching = [
            row.cid for row in rows
            if all(part in row.case["question"].casefold() for part in fragments)
        ]
        need(len(matching) == 1, "Motivating exposed case is no longer unique")
        first.extend(matching)
    # 3. Ordinal 25 (rank-1 latent blocked by old margin>0), 4. Ordinal 0 (previously invalid 5-frame citation)
    for idx in (25, 0):
        cid = rows[idx].cid
        if cid not in first:
            first.append(cid)
    return first + [row.cid for row in rows if row.cid not in first]


def layouts(d, row, processor, rope, foil):
    reports = row.requests
    grid, = {tuple(g) for g in reports[0]["image_grid_thw"]}
    args = row.old_text["arguments"] if row.old_text["valid"] else None
    query = args["query"] if args else row.case["question"]
    texts = [x["text"] for x in reports]
    cap = reports[0].get("max_input_tokens", 20000)
    pair = [
        receiver.render(d.s, processor, rope, grid, target_from_question(question), query, texts, cap)[0]
        for question in (row.case["question"], foil["question"])
    ]
    need(pair[0]["context_sha256"] == pair[1]["context_sha256"]
         and pair[0]["target_sha256"] != pair[1]["target_sha256"]
         and pair[0]["labels"] == pair[1]["labels"] and pair[0]["slot_tokens"] == pair[1]["slot_tokens"],
         "Receiver layout no longer preserves the matched input")
    return pair


def preflight():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preflight needs hidden GPUs")
    old, d, order = load_history()
    import run_itt_utility as native

    s = d.s
    s.torch.set_num_threads(2)
    model_path = s.old.n.MODEL
    processor = s.prepare_native(s.AutoProcessor.from_pretrained(model_path, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(model_path, local_files_only=True))
    foils = plan_foils(d.rows, processor.tokenizer)
    split, _, _, _ = old.bridge.attach_code()
    audited_contracts = 0
    for row in d.rows:
        layouts(d, row, processor, rope, foils[row.cid])
        geometry, sample, reference = old.bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
        need(reference.is_file(), "Native Focus reference missing")
        raw = row.old_text["arguments"] if row.old_text["valid"] else None
        if raw is None:
            multi_action.action_contract(geometry, sample, None, {}, dict(stage="fallback"))
            continue
        base = sample(raw["start_time"], raw["end_time"])
        draft = dict(frames={"text": base})
        multi_action.action_contract(geometry, sample, raw, draft, dict(stage="fallback"))
        if len(base) == 32:
            first, menu = multi_action.gap_pools(geometry["source"], geometry["fps"], base, sample)
            if menu:
                draft.update(first=first, menu=menu)
                seen_init = set(geometry["source"]) | set(base)
                ordered_keys = sorted(menu)
                for k in multi_action.ALLOWED_K:
                    if len(ordered_keys) < k:
                        continue
                    per_gap = 16 // k
                    s_addrs = policy._pick_nms_gaps(ordered_keys, k)
                    c_addrs = list(policy.select_clock_gaps(
                        menu, k, seen_init, multi_action.evenly_native, split.distance_sorted,
                    ))
                    s_extra = [f for a in s_addrs for f in multi_action.evenly_native(menu[a], per_gap)]
                    c_extra = [f for a in c_addrs for f in multi_action.evenly_native(menu[a], per_gap)]
                    selected = dict(
                        stage="passed", addresses=s_addrs, clock_addresses=c_addrs,
                        per_gap_frames=per_gap, frames=sorted(first + s_extra), clock=sorted(first + c_extra),
                    )
                    multi_action.action_contract(geometry, sample, raw, draft, selected)
                    audited_contracts += 1
    need(not s.torch.cuda.is_initialized(), "CPU preflight initialized CUDA")
    record = dict(
        status="entropy_routing_preflight", files=files(), order=order,
        execution_order=engineering_order(d.rows), foils=foils,
        bound_multiclip_contracts=audited_contracts,
        labels=0, holdout=0, gpu=0, real_model_calls=0,
    )
    pin = write(PREFLIGHT, record)
    emit(status=record["status"], videos=len(order), contracts=audited_contracts, pin=pin)


def execution(pin, uuid):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == uuid and uuid.startswith("GPU-")
         and os.environ.get("PYTORCH_CUDA_ALLOC_CONF") == "backend:native", "Expose exactly one A40")
    need(digest(PREFLIGHT) == pin, "Changed preflight receipt")
    manifest = json.loads(PREFLIGHT.read_bytes())
    need(manifest["files"] == files() and manifest["labels"] == manifest["holdout"] == manifest["gpu"] == 0,
         "Changed deployment after CPU preflight")
    old, d, order = load_history()
    need(order == manifest["order"] and engineering_order(d.rows) == manifest["execution_order"],
         "Changed exposed execution scope")
    import decord
    import focus_helpers as helper
    import run_itt_utility as native
    from live_dev import media_path

    t = d.s.torch
    with old.GUARD.open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        name, free, util = map(str.strip, old.smi(uuid, "--query-gpu=name,memory.free,utilization.gpu").split(","))
        need(name == "NVIDIA A40" and int(free) >= 36864 and int(util) <= 15
             and not old.smi(uuid, "--query-compute-apps=pid"), "Selected A40 is in use")
        d.s.old.verify_weights()
        need(t.cuda.device_count() == 1 and not t.cuda.memory_reserved() and t.cuda.get_allocator_backend() == "native",
             "Unexpected initial CUDA context")
        t.cuda.set_per_process_memory_fraction(old.CAP / t.cuda.get_device_properties(0).total_memory)
        peak = 0

        def occupancy():
            nonlocal peak
            peak = max(peak, t.cuda.max_memory_reserved())
            need(set(old.smi(uuid, "--query-compute-apps=pid").split()) <= {str(os.getpid())}
                 and peak <= old.CAP, "GPU occupancy or allocation changed")

        def reader(path):
            return decord.VideoReader(str(path), ctx=decord.cpu(0), num_threads=1)

        def stop(*_):
            raise InterruptedError("Entropy LAVE experiment stopped")

        signal.signal(signal.SIGTERM, stop)
        ACTION.mkdir(mode=0o700, exist_ok=False)
        totals, committed = Counter(), {}
        index = {row.cid: (i, row) for i, row in enumerate(d.rows)}
        terminal = dict(order=order, pin=pin, gpu=uuid, cards=1, labels=0, training=0, holdout=0, pid=os.getpid())
        emit(status="gpu_boot", pid=os.getpid(), cap_bytes=old.CAP)
        try:
            with (ROOT / "native.log").open("x", buffering=1) as sink:
                with redirect_stdout(sink):
                    model = d.s.boot(ACTION, uuid, d.nat)
                occupancy()
                foils = plan_foils(d.rows, model.processor.tokenizer)
                need(foils == manifest["foils"], "CPU and live tokenizer disagree on wrong-question controls")
                versions = tuple(p._version for p in model.model.parameters())
                for cid in manifest["execution_order"]:
                    ordinal, row = index[cid]
                    work = ACTION / f"case_{ordinal:02d}"
                    emit(status="case_start", ordinal=ordinal, cases=len(committed))
                    start = perf_counter()
                    result = run_live_case(
                        d, row, model, native, ordinal, work, foils[cid]["question"], occupancy, sink,
                        bridge=old.bridge, helper=helper, media_path=media_path, reader_factory=reader,
                        live_sources=old.live_sources, text_action=old.text_action,
                    )
                    trace = result["route_audit"]
                    focus_calls = sum(x["physical_calls"] for x in result["focus"].values())
                    generated = [
                        cell for cell in result["cells"].values()
                        if cell["targeted"]["status"] == "observed"
                    ]
                    k_used = len(trace["addresses"]) if trace and trace["addresses"] else 0
                    totals.update(
                        source_calls=2, planner_calls=1, focus_calls=focus_calls,
                        unique_qa=result["unique_qa_calls"], invalid_action=int(result["text_action"] is None),
                        scan=int(trace is not None),
                        forward_calls=0 if trace is None else (
                            trace["scan"]["forwards"] + (trace["controls"] or {}).get("forwards", 0)
                        ),
                        approved=int(k_used > 0),
                        targeted_cited_cells=len(generated),
                        **({f"k_{k_used}": 1} if k_used > 0 else {}),
                    )
                    committed[cid] = write(work / "case.json", dict(
                        cid=cid, ordinal=ordinal, case_uid=row.case["video_uid"],
                        elapsed_seconds=perf_counter() - start, foil_id=foils[cid]["foil_id"], **result,
                    ))
                    emit(status="case_committed", ordinal=ordinal, cases=len(committed),
                         k_used=k_used, peak_bytes=peak, **totals)
                    if len(committed) == 4:
                        need(totals["scan"] > 0 and totals["focus_calls"] > 0 and totals["unique_qa"] > 0,
                             "The engineering prefix did not exercise the complete entropy-LAVE chain")
                        emit(status="engineering_prefix_passed", cases=4, peak_bytes=peak, **totals)
                need(tuple(p._version for p in model.model.parameters()) == versions, "Model weights changed")
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
        execution(*args)
    else:
        raise SystemExit("preflight | run PIN UUID")
