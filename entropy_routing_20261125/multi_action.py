"""Exact multi-clip (1 + K) 32-frame action contract and counterbalanced triad audit."""

from hashlib import sha256
from itertools import permutations
import json
from math import isfinite


BUDGET = 32
ARMS = ("text", "state_split", "clock_split")
SCHEDULES = tuple(permutations(ARMS))
ALLOWED_K = (1, 2, 4)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def pin(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    return sha256(raw.encode()).hexdigest()


def order_for(case_ordinal):
    ensure(type(case_ordinal) is int and case_ordinal >= 0, "Ordinal must come from the frozen cohort")
    return SCHEDULES[case_ordinal % len(SCHEDULES)]


def evenly_native(values, k):
    ensure(type(k) is int and 1 <= k <= len(values), "Not enough distinct indices for allocation")
    if k == 1:
        return [values[len(values) // 2]]
    step = k - 1
    return [values[(i * (len(values) - 1) + step // 2) // step] for i in range(k)]


def _pool(window, sample):
    keys = {"start_time", "end_time", "frames"}
    ensure(type(window) is dict and set(window) == keys, "Operation needs exact public clip fields")
    lo, hi = window["start_time"], window["end_time"]
    ensure(all(type(v) in (int, float) and isfinite(v) for v in (lo, hi)) and 0 <= lo < hi,
           "Invalid physical interval")
    values = sample(lo, hi)
    ensure(type(values) is list and values == sorted(set(values)) and len(values) <= BUDGET
           and all(type(i) is int for i in values), "Unexpected Focus sampling grid")
    return values


def expand(action, sample):
    """Expand either one full 32-frame clip or 16-frame anchor + K equal gap clips totaling 32 frames."""
    ensure(type(action) is dict and set(action) == {"query", "clips"}
           and type(action["query"]) is str and action["query"].strip(), "Invalid Tool action")
    clips = action["clips"]
    ensure(type(clips) is list and len(clips) in (1, 2, 3, 5), "Allow 1 full clip or 1 anchor + K in {1,2,4} gap clips")
    ensure(all(type(w["frames"]) is int and w["frames"] > 0 for w in clips), "Use positive integer frame counts")
    pools = [_pool(w, sample) for w in clips]
    if len(clips) == 1:
        ensure(clips[0]["frames"] == BUDGET and pools[0], "Invalid ordinary allocation")
        return pools[0]

    k = len(clips) - 1
    ensure(k in ALLOWED_K, "Unsupported multi-gap split count")
    per_gap = 16 // k
    ensure(clips[0]["frames"] == 16 and len(pools[0]) == BUDGET
           and all(w["frames"] == per_gap for w in clips[1:]),
           "Split clips must retain 16 anchor frames plus K equal gap allocations totaling 16")
    first = evenly_native(pools[0], 16)
    taken = set(first)
    extra = []
    for clip_pool in pools[1:]:
        remain = sorted(set(clip_pool) - taken)
        picked = evenly_native(remain, per_gap)
        extra.extend(picked)
        taken.update(picked)
    indices = sorted(first + extra)
    ensure(len(indices) == len(set(indices)) == BUDGET, "Multi-clip frame allocation must remain exact 32")
    return indices


def gap_pools(source, fps, base, sample):
    """Return the 16-frame text anchor and exact unseen interior frame pools for every legal Source gap."""
    ensure(len(source) == 32 and sorted(set(source)) == source and isfinite(fps) and fps > 0, "Invalid Source geometry")
    ensure(len(base) == 32 and sorted(set(base)) == base, "Invalid strong-text anchor")
    first = evenly_native(base, 16)
    seen = set(source) | set(first)
    menu = {}
    for i, (left, right) in enumerate(zip(source, source[1:])):
        available = sorted({
            frame for frame in sample(left / fps, right / fps)
            if left < frame < right and frame not in seen
        })
        if len(available) >= 16:
            menu[i] = available
    return first, menu


def bind(action_row, sample, archived_text):
    """Require exact arguments -> indices reconstruction and matched K between state_split and clock_split."""
    ensure(set(action_row) == {"arguments", "indices"}
           and set(action_row["arguments"]) == set(action_row["indices"]) == set(ARMS), "Changed arm roster")
    act = action_row["arguments"]
    old = act["text"]
    physical, keys = {}, {}
    ensure((old is None) == (archived_text is None), "Original common skip changed")
    if old is None:
        ensure(all(act[a] is None and action_row["indices"][a] == [] for a in ARMS),
               "Split arms cannot rescue an invalid text action")
        return physical, keys
    text = [dict(start_time=archived_text["start_time"], end_time=archived_text["end_time"], frames=BUDGET)]
    ensure(old == dict(query=archived_text["query"], clips=text), "Strong text anchor differs from original Tool plan")
    ensure(all(a is not None for a in act.values()), "A legal comparison needs all three arms")
    changed = act["state_split"] != old or act["clock_split"] != old
    if changed:
        first = dict(text[0], frames=16)
        s_clips, c_clips = act["state_split"]["clips"], act["clock_split"]["clips"]
        ensure(len(s_clips) == len(c_clips) and len(s_clips) - 1 in ALLOWED_K
               and s_clips[0] == first and c_clips[0] == first
               and [w["frames"] for w in s_clips] == [w["frames"] for w in c_clips],
               "State and clock splits must share the same 16-frame anchor, K, and per-clip frame counts")
    for arm in ARMS:
        route = act[arm]
        ensure(route["query"] == old["query"], "A split changed the original query")
        indices = expand(route, sample)
        ensure(indices == action_row["indices"][arm], "Native multi-clip reconstruction differs from sealed request")
        physical[arm] = indices
        keys[arm] = pin(dict(query=route["query"], frames=indices, header="joint_multi_v1"))
    return physical, keys


def action_contract(geometry, sample, text, draft, selected):
    """Build exact multi-clip arguments and bounded sampler for all three matched arms."""
    if text is None:
        action = dict(arguments=dict.fromkeys(ARMS), indices={a: [] for a in ARMS})
        bind(action, sample, text)
        return action, sample
    first_full = dict(start_time=text["start_time"], end_time=text["end_time"], frames=BUDGET)
    ordinary = dict(query=text["query"], clips=[first_full])
    if selected["stage"] != "passed":
        action = dict(arguments=dict.fromkeys(ARMS, ordinary),
                      indices=dict.fromkeys(ARMS, draft["frames"]["text"]))
        bind(action, sample, text)
        return action, sample

    anchors, fps = geometry["source"], geometry["fps"]
    source_set = set(anchors)
    per_gap = selected["per_gap_frames"]
    first_half = dict(first_full, frames=16)
    ordinary_range = (text["start_time"], text["end_time"])
    altered, pools = {}, {}
    for name, addresses in (("state_split", selected["addresses"]), ("clock_split", selected["clock_addresses"])):
        clips = [first_half]
        for address in addresses:
            left, right = anchors[address:address + 2]
            lo, hi = left / fps, right / fps
            clips.append(dict(start_time=lo, end_time=hi, frames=per_gap))
            pools[(lo, hi)] = [k for k in sample(lo, hi) if left < k < right and k not in source_set]
        altered[name] = dict(query=text["query"], clips=clips)
    pools.pop(ordinary_range, None)

    def bounded_sampler(lo, hi):
        selected_pool = pools.get((lo, hi))
        return sample(lo, hi) if selected_pool is None else selected_pool

    action = dict(
        arguments=dict(text=ordinary, **altered),
        indices=dict(text=draft["frames"]["text"], state_split=selected["frames"], clock_split=selected["clock"]),
    )
    bind(action, bounded_sampler, text)
    return action, bounded_sampler


def audit(result, action_row, case_ordinal, sample, archived_text):
    """Recompute canonical calls and exact-input reuse from the frozen multi-clip action."""
    ensure(set(result) == {"rows", "execution_order", "physical_calls", "bindings"}, "Unexpected acquisition record")
    sequence = order_for(case_ordinal)
    ensure(tuple(result["execution_order"]) == sequence and set(result["rows"]) == set(ARMS),
           "Missing arms or changed counterbalanced order")
    physical, keys = bind(action_row, sample, archived_text)
    ensure(result["bindings"] == keys, "Observation binding changed")
    seen, calls = {}, 0
    for arm in sequence:
        row = result["rows"][arm]
        if not keys:
            ensure(row == dict(status="common_invalid_action", calls=0, observation=None),
                   "Common fallback must not generate")
            continue
        key = keys[arm]
        if key in seen:
            source = seen[key]
            expect = dict(result["rows"][source], calls=0, reused_identical_input=source)
            ensure(row == expect, "Only exact same-session observation input can share a generation")
            continue
        fields = {"status", "calls", "indices", "clocks", "jpeg_sha256", "rgb_sha256",
                  "observation", "capped", "record"}
        ensure(set(row) == fields and row["status"] == "focus_generated" and type(row["calls"]) is int
               and row["calls"] == 1 and row["indices"] == physical[arm]
               and type(row["observation"]) is str and type(row["capped"]) is bool
               and len(row["indices"]) == len(row["clocks"]) == len(row["jpeg_sha256"]) == len(row["rgb_sha256"])
               and isinstance(row["record"], dict) and row["record"],
               "Native acquisition did not execute the sealed frame request")
        calls += 1
        seen[key] = arm
    ensure(calls == result["physical_calls"] == len(set(keys.values())), "Incorrect call budget")
    return calls


def execute(action_row, case_ordinal, sample, archived_text, observe):
    """Execute once per physically distinct frame set in counterbalanced order."""
    sequence = order_for(case_ordinal)
    physical, keys = bind(action_row, sample, archived_text)
    outcomes, seen = {}, {}
    for arm in sequence:
        if not keys:
            outcomes[arm] = dict(status="common_invalid_action", calls=0, observation=None)
            continue
        key = keys[arm]
        if key in seen:
            source = seen[key]
            outcomes[arm] = dict(outcomes[source], calls=0, reused_identical_input=source)
        else:
            outcomes[arm] = observe(arm, physical[arm], archived_text)
            seen[key] = arm
    result = dict(rows=outcomes, execution_order=list(sequence), physical_calls=len(seen), bindings=keys)
    audit(result, action_row, case_ordinal, sample, archived_text)
    return result
