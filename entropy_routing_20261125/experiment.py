"""Bind calibrated-lift entropy multi-gap routing and matched 3x2 Focus evaluation."""

from copy import deepcopy
from dataclasses import asdict
import json
from math import isfinite

from missing_fact_routing_20261124 import receiver
from missing_fact_routing_20261124.target import needs_verification

from . import multi_action, policy
from .report import render_focus_prompt, serialize_focus_report


MODES = ("native", "targeted")
NATIVE_REPORT_INSTRUCTION = "\n\nPlease answer the question based on the given video clip. "


def select_addresses(d, row, vlm, states, target, wrong_target, text, reports, eligible):
    """Return a route audit with calibrated-lift controls and entropy-selected gap addresses."""
    grid, = {tuple(size) for size in reports[0]["image_grid_thw"]}
    config, processor = vlm.model.config, vlm.processor
    rope = d.s.rope(config)
    cap = row.requests[0].get("max_input_tokens", 20000)
    history = [record["text"] for record in reports]
    actual = receiver.render(d.s, processor, rope, grid, target, text["query"], history, cap)
    scan = receiver.probe_states(d, vlm, states, actual)
    candidate = policy.nominate(scan["real"], scan["zero"], eligible)
    audit = dict(scan=scan, controls=None, decision=None, addresses=(), per_gap_frames=0,
                 actual_prompt=actual[0], wrong_prompt=None)
    if candidate is not None:
        foil = receiver.render(d.s, processor, rope, grid, wrong_target, text["query"], history, cap)
        control = receiver.probe_controls(d, row, vlm, states, candidate.slot, actual, foil)
        decision = policy.approve(candidate, scan["real"], scan["zero"], control["scrambled"],
                                  control["wrong"], control["wrong_zero"], eligible)
        audit.update(
            controls=control,
            decision=dict(asdict(decision), addresses=list(decision.addresses)),
            wrong_prompt=foil[0],
            addresses=list(decision.addresses) if decision.accepted else [],
            per_gap_frames=decision.per_gap_frames if decision.accepted else 0,
        )
    else:
        audit["addresses"] = []
    return audit


def plan_actions(split, text, geometry, sample, choose, *, source_status=None):
    """Build exact 32-frame multi-clip requests for text, entropy-routed latent, and matched clock."""
    selected = dict(stage="fallback")
    if text is None:
        action, sampler = multi_action.action_contract(geometry, sample, None, {}, selected)
        return action, sampler, None
    if not isinstance(text.get("query"), str) or not text["query"].strip():
        raise ValueError("Expected a validated native Tool action")
    base = sample(text["start_time"], text["end_time"])
    draft = dict(frames={"text": base})
    audit = None
    if len(base) == 32 and needs_verification(source_status):
        first, menu = multi_action.gap_pools(geometry["source"], geometry["fps"], base, sample)
        if menu:
            audit = choose(sorted(menu))
            addresses = tuple(audit["addresses"])
            if addresses:
                k = len(addresses)
                per_gap = audit["per_gap_frames"]
                if (k not in multi_action.ALLOWED_K or per_gap != 16 // k
                        or len(set(addresses)) != k or any(a not in menu for a in addresses)):
                    raise ValueError("Receiver chose invalid multi-gap addresses")
                seen_init = set(geometry["source"]) | set(base)
                clock_addresses = policy.select_clock_gaps(
                    menu, k, seen_init, multi_action.evenly_native, split.distance_sorted,
                )
                state_extra = [f for a in addresses for f in multi_action.evenly_native(menu[a], per_gap)]
                clock_extra = [f for a in clock_addresses for f in multi_action.evenly_native(menu[a], per_gap)]
                draft.update(menu=menu, first=first)
                selected = dict(
                    stage="passed",
                    addresses=list(addresses),
                    clock_addresses=list(clock_addresses),
                    per_gap_frames=per_gap,
                    frames=sorted(first + state_extra),
                    clock=sorted(first + clock_extra),
                )
    action, sampler = multi_action.action_contract(geometry, sample, text, draft, selected)
    return action, sampler, audit


def targeted_view(rendered, target, fps):
    """Replace only the reporting instructions within the exact visual request."""
    if type(fps) not in (int, float) or not isfinite(fps) or fps <= 0:
        raise ValueError("Expected the video frame rate")
    times = [frame / fps for frame in rendered["indices"]]
    directed = render_focus_prompt(target, times)[0]["content"]
    labels = "\n".join(item["text"] for item in directed[:-1:2])
    compiled = deepcopy(rendered)
    final = compiled["messages"][0]["content"][-1]
    request, separator, _ = final["text"].rpartition(NATIVE_REPORT_INSTRUCTION)
    if not separator:
        raise ValueError("Missing native report instruction boundary")
    final["text"] = (
        request + "\n\nUse these integer frame IDs for the samples above, in their displayed order:\n"
        + labels + "\n\n" + directed[-1]["text"]
    )
    return compiled


def run_factorial(io, native, case, reports, target, fps, ordinal, action, sampler, text, observe, answer):
    """Run native and multi-frame targeted report modes against identical frames, then pure-text QA."""
    if target.question != case["question"] or len(reports) != 2:
        raise ValueError("The experiment must preserve the original question and both Source reports")
    order = MODES if (ordinal // 6) % 2 == 0 else MODES[::-1]
    ledgers = {
        mode: multi_action.execute(action, ordinal, sampler, text,
                                   lambda arm, indices, query: observe(mode, arm, indices, query))
        for mode in order
    }
    arms = multi_action.order_for(ordinal)
    for arm in arms:
        old = ledgers["native"]["rows"][arm]
        new = ledgers["targeted"]["rows"][arm]
        for field in ("indices", "clocks", "jpeg_sha256", "rgb_sha256"):
            if old.get(field) != new.get(field):
                raise ValueError("Report modes received different visual evidence")

    outputs = {}
    previous_qa = {}
    for arm in arms:
        outputs[arm] = {}
        for mode in order:
            row = ledgers[mode]["rows"][arm]
            observation = row["observation"]
            if observation is None:
                outputs[arm][mode] = dict(status="common_invalid", observation=None, qa=None)
                continue
            status = "native"
            if mode == "targeted":
                times = [frame / fps for frame in row["indices"]]
                status, observation = serialize_focus_report(observation, times, capped=row["capped"])
            messages = io.qa_messages(native, case, reports, observation)
            key = json.dumps(messages, ensure_ascii=False, sort_keys=True, allow_nan=False)
            reused_from = None
            if key in previous_qa:
                origin_mode, origin_arm, answer_record = previous_qa[key]
                reused_from = dict(mode=origin_mode, arm=origin_arm)
            else:
                answer_record = answer(mode, arm, messages)
                previous_qa[key] = mode, arm, answer_record
            outputs[arm][mode] = dict(status=status, observation=observation,
                                      capped=row["capped"], qa=answer_record, qa_reused_from=reused_from)
    return dict(target=asdict(target), focus=ledgers, cells=outputs, unique_qa_calls=len(previous_qa))
