"""Bind missing-fact readout and evaluate a matched 3-by-2 Focus design."""

from copy import deepcopy
from dataclasses import asdict
import json
from math import isfinite

from . import policy, receiver
from .report import render_focus_prompt, serialize_focus_report
from .target import needs_verification


MODES = ("native", "targeted")
NATIVE_REPORT_INSTRUCTION = "\n\nPlease answer the question based on the given video clip. "


def select_address(d, row, vlm, states, target, wrong_target, text, reports, eligible):
    """Return a route audit with only a time address approved for the Tool."""
    grid, = {tuple(size) for size in reports[0]["image_grid_thw"]}
    config, processor = vlm.model.config, vlm.processor
    rope = d.s.rope(config)
    cap = row.requests[0].get("max_input_tokens", 20000)
    history = [record["text"] for record in reports]
    actual = receiver.render(d.s, processor, rope, grid, target, text["query"], history, cap)
    scan = receiver.probe_states(d, vlm, states, actual)
    candidate = policy.nominate(scan["real"], scan["zero"], eligible)
    audit = dict(scan=scan, controls=None, decision=None, address=None,
                 actual_prompt=actual[0], wrong_prompt=None)
    if candidate is not None:
        foil = receiver.render(d.s, processor, rope, grid, wrong_target, text["query"], history, cap)
        control = receiver.probe_controls(d, row, vlm, states, candidate.slot, actual, foil)
        decision = policy.approve(candidate, scan["zero"], control["scrambled"],
                                  control["wrong"], control["wrong_zero"])
        audit.update(controls=control, decision=asdict(decision), wrong_prompt=foil[0],
                     address=candidate.address if decision.accepted else None)
    return audit


def plan_actions(split, triad, action_contract, gaps, text, geometry, sample, choose, *, source_status=None):
    """Use the archived timing contract and clock ranking for a matched triplet."""
    selected = dict(stage="fallback")
    if text is None:
        action, sampler = action_contract(split, triad, geometry, sample, None, {}, selected)
        return action, sampler, None
    if not isinstance(text.get("query"), str) or not text["query"].strip():
        raise ValueError("Expected a validated native Tool action")
    archived_range = text["start_time"], text["end_time"]
    base = sample(*archived_range)
    draft = dict(frames={"text": base})
    audit = None
    if len(base) == 32 and needs_verification(source_status):
        first, menu = gaps(geometry["source"], geometry["fps"], base, sample, split.evenly_native)
        if menu:
            audit = choose(sorted(menu))
            address = audit["address"]
            if address is not None:
                if type(address) is not int or address not in menu:
                    raise ValueError("The receiver chose an address outside the fixed menu")
                seen = sorted(set(geometry["source"]) | set(base))

                def novelty(k):
                    distances = sorted(split.distance_sorted(i, seen) for i in menu[k])
                    return distances[7], sum(distances), -k

                clock = max(menu, key=novelty)
                draft.update(menu=menu, first=first)
                selected = dict(stage="passed", address=address,
                                frames=sorted(first + menu[address]), clock=sorted(first + menu[clock]))
    action, sampler = action_contract(split, triad, geometry, sample, text, draft, selected)
    return action, sampler, audit


def targeted_view(rendered, target, fps):
    """Replace the reporting instructions within the original visual request."""
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
        request + "\n\nUse these frame IDs for the samples above, in their displayed order:\n"
        + labels + "\n\n" + directed[-1]["text"]
    )
    return compiled


def run_factorial(triad, io, native, case, reports, target, fps, ordinal, action, sampler, text, observe, answer):
    """Run the two report modes against the same native frame action, then text-only QA."""
    if target.question != case["question"] or len(reports) != 2:
        raise ValueError("The experiment must preserve the original question and both Source reports")
    order = MODES if (ordinal // 6) % 2 == 0 else MODES[::-1]
    ledgers = {
        mode: triad.execute(action, ordinal, sampler, text,
                            lambda arm, indices, query: observe(mode, arm, indices, query))
        for mode in order
    }
    arms = triad.order_for(ordinal)
    for arm in arms:
        old = ledgers["native"]["rows"][arm]
        new = ledgers["targeted"]["rows"][arm]
        key_fields = ("indices", "clocks", "jpeg_sha256", "rgb_sha256")
        if any(old.get(field) != new.get(field) for field in key_fields):
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
