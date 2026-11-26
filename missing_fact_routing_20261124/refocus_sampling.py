"""Propose two disjoint eight-frame additions to a previously sampled Focus view."""

from bisect import bisect_left, insort
from math import floor, isfinite


FRAME_BUDGET = 8
RADIUS_SECONDS = 2


def _frames(values, frame_count):
    frames = tuple(values)
    if any(type(frame) is not int or not 0 <= frame < frame_count for frame in frames):
        raise ValueError("Frame indices must be integers inside the sealed video")
    return set(frames)


def _union(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(hi, merged[-1][1]))
        else:
            merged.append((lo, hi))
    return merged


def _available(intervals, excluded):
    available = []
    for lo, hi in intervals:
        cursor = lo
        for left, right in excluded:
            if cursor > hi or left > hi:
                break
            if cursor < left:
                available.append((cursor, left - 1))
            cursor = max(cursor, right + 1)
        if cursor <= hi:
            available.append((cursor, hi))
    return available


def _coverage(focus, clips, fps, frame_count):
    if clips is None:
        return [(min(focus), max(focus))]
    clips = tuple(clips)
    if not 1 <= len(clips) <= 2:
        raise ValueError("Expected one or two original Focus clips")
    ranges = []
    for clip in clips:
        start, end = clip["start_time"], clip["end_time"]
        if any(type(t) not in (int, float) or not isfinite(t) for t in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid original Focus time bounds")
        # The native Focus samples range(left, right), with right at most count - 1.
        lo, hi = floor(min(start * fps, frame_count)), floor(min(end * fps, frame_count - 1)) - 1
        if lo > hi:
            raise ValueError("An original Focus clip has no sampleable frames")
        ranges.append((lo, hi))
    if any(not any(lo <= frame <= hi for lo, hi in ranges) for frame in focus):
        raise ValueError("A sealed Focus frame lies outside the original clips")
    return _union(ranges)


def _local_sample(cited, radius, intervals):
    remaining = {frame for lo, hi in intervals for frame in range(lo, hi + 1)}
    chosen = []
    per_side = FRAME_BUDGET // (2 * len(cited))
    # Equal slots before and after each citation, with evenly spaced offsets.
    for step in range(per_side):
        offset = (2 * step + 1) * radius / (2 * per_side)
        for anchor in cited:
            for side in (-1, 1):
                legal = (i for i in remaining if 0 < side * (i - anchor) <= radius)
                i = min(legal, key=lambda i: (abs(i - anchor - side * offset), i), default=None)
                if i is not None:
                    chosen.append(i)
                    remaining.remove(i)
    # A clipped or overlapping neighborhood can exhaust one side. Spread its
    # unused slots across the remaining local evidence rather than leave holes.
    while len(chosen) < FRAME_BUDGET:
        known = (*cited, *chosen)
        i = max(remaining, key=lambda j: (min(abs(j - seen) for seen in known), -j))
        chosen.append(i)
        remaining.remove(i)
    return sorted(chosen)


def _clock_sample(intervals, previously_seen):
    available, known, chosen = list(intervals), sorted(previously_seen), []
    while len(chosen) < FRAME_BUDGET:
        options = []
        for j, (lo, hi) in enumerate(available):
            pos = bisect_left(known, lo)
            if pos == 0:
                index, distance = lo, known[0] - lo
            elif pos == len(known):
                index, distance = hi, hi - known[-1]
            else:
                left, right = known[pos - 1], known[pos]
                index = min(max((left + right) // 2, lo), hi)
                distance = min(index - left, right - index)
            options.append((distance, -index, j))
        _, index, j = max(options)
        index = -index
        lo, hi = available.pop(j)
        available.extend((a, b) for a, b in ((lo, index - 1), (index + 1, hi)) if a <= b)
        insort(known, index)
        chosen.append(index)
    return sorted(chosen)


def plan_refocus(*, cited_indices, source_indices, focus_indices, fps, frame_count,
                 focus_clips=None, other_focus_indices=()):
    """Plan local and clock additions from frozen frame indices only.

    Local frames are within two seconds of an original citation. Clock frames
    are outside those neighborhoods and inside the original clip union. With
    no clip bounds supplied, the earliest and latest Focus indices bound that
    coverage. Clock greedily maximizes the distance to any previously seen or
    already selected control frame, breaking ties by earlier index.
    """
    if type(frame_count) is not int or frame_count < 1 or type(fps) not in (int, float) or not isfinite(fps) or fps <= 0:
        raise ValueError("Expected actual positive FPS and frame count")
    anchors = tuple(cited_indices)
    cited = sorted(_frames(anchors, frame_count))
    focus = _frames(focus_indices, frame_count)
    if len(cited) != len(anchors) or not 1 <= len(cited) <= 2 or not set(cited) <= focus:
        raise ValueError("Expected one or two distinct original Focus citation indices")
    seen = focus | _frames(source_indices, frame_count) | _frames(other_focus_indices, frame_count)
    coverage = _coverage(focus, focus_clips, fps, frame_count)
    radius = floor(min(RADIUS_SECONDS * fps, frame_count - 1))
    neighborhoods = _union((max(0, c - radius), min(frame_count - 1, c + radius)) for c in cited)
    seen_ranges = _union((frame, frame) for frame in seen)
    local = _available(neighborhoods, seen_ranges)
    clock = _available(coverage, _union([*neighborhoods, *seen_ranges]))
    local_count = sum(hi - lo + 1 for lo, hi in local)
    clock_count = sum(hi - lo + 1 for lo, hi in clock)
    missing = [name for name, count in (("local", local_count), ("clock", clock_count)) if count < FRAME_BUDGET]
    return dict(
        status="no_plan" if missing else "planned", reason="insufficient_" + "_and_".join(missing) if missing else None,
        cited_indices=cited, local_radius_seconds=RADIUS_SECONDS,
        local_frame_ranges=[list(bounds) for bounds in neighborhoods],
        coverage_frame_ranges=[list(bounds) for bounds in coverage],
        local_pool_size=local_count, clock_pool_size=clock_count,
        local_indices=[] if missing else _local_sample(cited, radius, local),
        clock_indices=[] if missing else _clock_sample(clock, seen),
    )
