"""Bind one development case to an already admitted native LAVE runtime."""

from hashlib import sha256
import json
from pathlib import Path

from .experiment import plan_actions, run_factorial, select_address
from .native_runtime import NativeAdapter
from .target import needs_verification, source_check_messages, target_from_question


def _historical_receipt(block):
    preflight = (block / "preflight.json").read_bytes()
    manifest = (block / "protocol_manifest.json").read_bytes()
    recorded = json.loads(preflight)
    if not recorded["passed"] or recorded["manifest_sha256"] != sha256(manifest).hexdigest():
        raise ValueError("Native LAVE manifest no longer matches its preflight")
    return dict(preflight=preflight, manifest=manifest, preflight_pin=sha256(preflight).hexdigest())


def run_live_case(
    d, row, model, native, ordinal, work, wrong_question, occupancy, sink, *,
    bridge, helper, media_path, reader_factory, action_contract, gaps,
    live_sources, text_action, check_source=False,
):
    """Use the native live Source and Tool planner, then compare three Focus frame routes.

    The caller admits the already loaded model and supplies the pinned bridge,
    label-blind development row, wrong question, and a CPU reader factory. The
    media_path callback must attest the video bytes against the row's seal.
    """
    if row.case["split"] != "development" or type(ordinal) is not int or ordinal < 0:
        raise ValueError("Expected an ordered development case")
    target = target_from_question(row.case["question"])
    wrong_target = target_from_question(wrong_question)
    if target.question.strip().casefold() == wrong_target.question.strip().casefold():
        raise ValueError("The wrong-question control must use a different question")

    work = Path(work)
    work.mkdir(mode=0o700, exist_ok=False)
    split, triad, runtime, io = bridge.attach_code()
    geometry, sample, reference = bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
    historical = _historical_receipt(row.job["block"])
    reports, states = live_sources(d, row, model, work, occupancy, sink)
    if len(reports) != 2 or any(not isinstance(record["text"], str) for record in reports):
        raise ValueError("The native Source must return two text reports")
    occupancy()
    text = text_action(d, row, model, work, reports, sink)
    occupancy()
    prior = tuple(record["text"] for record in reports)
    reader = reader_factory(media_path(row)) if text is not None else None
    adapter = NativeAdapter(
        native=native, io=io, helper=helper, reference=reference,
        joint_render=triad, joint_bridge=triad, runtime=runtime, historical=historical,
        work=work / "followup", model=model, reader=reader, fps=geometry["fps"],
        count=geometry["count"], case=row.case, reports=prior, target=target,
        occupancy=occupancy, sink=sink,
    )

    source_check = None
    if check_source and text is not None:
        measurement = adapter.check_source(source_check_messages(target, prior))
        status = "VERIFY" if measurement["capped"] or needs_verification(measurement["text"]) else "RESOLVED"
        source_check = dict(status=status, **measurement)

    def choose(eligible):
        occupancy()
        result = select_address(d, row, model, states, target, wrong_target, text, reports, eligible)
        occupancy()
        return result

    action, sampler, route_audit = plan_actions(
        split, triad, action_contract, gaps, text, geometry, sample, choose,
        source_status=source_check["status"] if source_check is not None else None,
    )
    del states
    evaluated = run_factorial(
        triad, io, native, row.case, prior, target, geometry["fps"], ordinal,
        action, sampler, text, adapter.observe, adapter.answer,
    )
    return dict(text_action=text, source_check=source_check, routes=action, route_audit=route_audit,
                visual_ledger=adapter.focus_ledger, **evaluated)
