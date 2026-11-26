"""Freeze an F1-state choice between equal before/after F2 frame requests."""

from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from math import floor, isfinite

from .policy import _log_odds
from .refocus_sampling import FRAME_BUDGET, RADIUS_SECONDS, _frames


SIDES = ("before", "after")
# Context can lie just outside a half-window but must still concern the local
# event. Two window widths also bound the gap when first-pass sampling is sparse.
MAX_CONTEXT_SECONDS = 2 * RADIUS_SECONDS


def _nearest_distance(index, seen):
    p = bisect_left(seen, index)
    return min(abs(index - seen[j]) for j in (p - 1, p) if 0 <= j < len(seen))


def plan_focus_sides(*, cited_indices, source_indices, focus_indices, fps, frame_count, other_focus_indices=()):
    """Freeze t−2s..t and t..t+2s requests from the earliest/latest citations.

    A multiple-frame citation defines one span, with both options outside it.
    Both windows use the same frame width and must fit the video without
    clipping. Each option samples the same eight quantile positions in its
    unseen frame pool. The real state for each option is the closest observed
    F1 slot on that side. Its shifted control is the exact half-turn in the
    original slot order, rounded down when the F1 frame count is odd.
    """
    if type(frame_count) is not int or frame_count < 1 or type(fps) not in (int, float) or not isfinite(fps) or fps <= 0:
        raise ValueError("Expected actual positive FPS and frame count")
    focus, anchors = list(focus_indices), list(cited_indices)
    if focus != sorted(_frames(focus, frame_count)):
        raise ValueError("F1 frames must be in exact chronological slot order")
    cited = sorted(_frames(anchors, frame_count))
    if len(cited) != len(anchors) or not 1 <= len(cited) <= 2 or not set(cited) <= set(focus):
        raise ValueError("Expected one or two distinct original F1 citation indices")

    plan = dict(status="no_plan", reason=None, menu={}, clock_side=None, cited_indices=cited,
                frame_budget=FRAME_BUDGET, half_window_seconds=RADIUS_SECONDS,
                max_context_seconds=MAX_CONTEXT_SECONDS, windows={}, pools={}, contexts={})

    width = floor(min(RADIUS_SECONDS * fps, frame_count))
    before = (cited[0] - width, cited[0] - 1)
    after = (cited[-1] + 1, cited[-1] + width)
    plan["windows"] = {side: list(bounds) for side, bounds in zip(SIDES, (before, after))}
    if before[0] < 0 or after[1] >= frame_count:
        return dict(plan, reason="half_window_crosses_video_boundary")

    seen = set(focus) | _frames(source_indices, frame_count) | _frames(other_focus_indices, frame_count)
    pools = {side: [i for i in range(lo, hi + 1) if i not in seen]
             for side, (lo, hi) in plan["windows"].items()}
    plan["pools"] = {side: len(pool) for side, pool in pools.items()}
    if any(len(pool) < FRAME_BUDGET for pool in pools.values()):
        return dict(plan, reason="insufficient_new_frames")

    context_slots = (bisect_left(focus, cited[0]) - 1, bisect_right(focus, cited[-1]))
    if context_slots[0] < 0 or context_slots[1] >= len(focus):
        return dict(plan, reason="missing_observed_f1_context")
    shifts = tuple((slot + len(focus) // 2) % len(focus) for slot in context_slots)
    deltas = (cited[0] - focus[context_slots[0]], focus[context_slots[1]] - cited[-1])
    plan["contexts"] = {
        side: dict(slot=slot, index=focus[slot], distance_frames=delta,
                   shifted_slot=shift, shifted_index=focus[shift])
        for side, slot, delta, shift in zip(SIDES, context_slots, deltas, shifts)
    }
    if any(delta / fps > MAX_CONTEXT_SECONDS for delta in deltas):
        return dict(plan, reason="f1_context_too_distant")
    if set(shifts) & set(context_slots) or any(focus[shift] in cited for shift in shifts):
        return dict(plan, reason="half_turn_aliases_context_or_citation")

    ordered = sorted(seen)
    clock_ranks = {}
    for j, (side, pool) in enumerate(pools.items()):
        frames = [pool[(2 * i + 1) * len(pool) // (2 * FRAME_BUDGET)] for i in range(FRAME_BUDGET)]
        distances = sorted(_nearest_distance(frame, ordered) for frame in frames)
        plan["menu"][side] = dict(frame_indices=frames, **plan["contexts"][side],
                                  nearest_seen_distance_frames=distances)
        clock_ranks[side] = (distances[FRAME_BUDGET // 2 - 1], sum(distances), -j)
    return dict(plan, status="planned", clock_side=max(clock_ranks, key=clock_ranks.get))


def _side_margins(pairs):
    if not isinstance(pairs, Mapping) or set(pairs) != set(SIDES):
        raise ValueError("Expected before/after Y/N logit pairs")
    return {side: _log_odds(pairs[side]) for side in SIDES}


def approve_side(plan, real, zero, shifted, scrambled, wrong, wrong_zero):
    """Accept a positive, unique real side only if it beats all content controls.

    True and wrong layouts have separate zero baselines. Control thresholds use
    each pair's strongest side. A tie or failed control uses the frozen clock
    side, while a physically unavailable menu remains skipped.
    """
    if plan["status"] == "no_plan":
        return dict(status="no_plan", reason=plan["reason"], nominated_side=None, selected_side=None)
    if plan["status"] != "planned" or plan["clock_side"] not in SIDES or set(plan["menu"]) != set(SIDES):
        raise ValueError("Expected the frozen two-side plan")
    blank, wrong_blank = _log_odds(zero), _log_odds(wrong_zero)
    margins = _side_margins(real)
    lifts = {side: margin - blank for side, margin in margins.items()}
    control = {
        "shifted": max(_side_margins(shifted).values()) - blank,
        "scrambled": max(_side_margins(scrambled).values()) - blank,
        "wrong_target": max(_side_margins(wrong).values()) - wrong_blank,
    }
    if any(not isfinite(v) for v in (*lifts.values(), *control.values())):
        raise ValueError("Nonfinite receiver lift")
    winner = max(lifts, key=lifts.get)
    other = "after" if winner == "before" else "before"
    nominated = lifts[winner] > max(0, lifts[other]) and margins[winner] > 0
    passed = nominated and lifts[winner] > max(control.values())
    return dict(status="approved" if passed else "fallback_clock", nominated_side=winner if nominated else None,
                selected_side=winner if passed else plan["clock_side"], clock_side=plan["clock_side"],
                real_margins=margins, real_lifts=lifts, control_lifts=control)
