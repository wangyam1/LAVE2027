"""Render a shared multi-frame Focus evidence request and validate bounded integer citations."""

from collections.abc import Mapping
import json
from math import isfinite
import re


UNRESOLVED_REPORT = "No relevant content found."
FIELDS = ("question", "entity", "attribute", "time_condition")
MAX_CITED_FRAMES = 8
MAX_OBSERVATION_CHARS = 480
INSTRUCTIONS = (
    "Inspect the sampled frames for the requested fact. Match the specified entity, "
    "attribute and time condition. Report only specific facts directly observable "
    "in the sampled frames. Keep any action, count or comparison scoped to the cited frames. "
    "Do not infer an unseen value, repeat the request as evidence, or give an option label. "
    "Respond with exactly one JSON object containing \"frames\" (an array of 1 to 6 distinct "
    "integer frame IDs from the Frame 0..N-1 labels above; never timestamps in seconds) and "
    "\"observation\" (one single-line string of one or two concise factual sentences covering "
    "all directly visible evidence for the requested fact). If these frames do not let you "
    "determine the requested fact, return UNRESOLVED."
)
FENCE = re.compile(r"```(?:json)?\s*([\s\S]+?)\s*```", re.IGNORECASE)
ANSWER_LABEL = re.compile(
    r"\b(?:option|choice)\s+(?:label\s+)?\(?[a-z]\)?\b"
    r"|\banswer\s*(?:is\s+|[:：]\s*)\(?[a-z]\)?\b"
    r"|(?:选项|答案)\s*(?:为|是|[:：])?\s*[(（]?[a-z][)）]?(?![a-z0-9])",
    re.IGNORECASE,
)


def _checked_times(frame_times):
    times = tuple(frame_times)
    if not times or any(type(t) not in (int, float) or not isfinite(t) or t < 0 for t in times):
        raise ValueError("Focus needs finite, nonnegative frame times in seconds")
    if any(first > second for first, second in zip(times, times[1:])):
        raise ValueError("Focus frame times must be chronological")
    return times


def _frame_label(index, times):
    return f"Frame {index} ({times[index]:.3f}s)"


def render_focus_prompt(target, frame_times):
    """Return the native user content format with one placeholder per sampled frame."""
    times = _checked_times(frame_times)
    values = {
        key: target.get(key) if isinstance(target, Mapping) else getattr(target, key, None)
        for key in FIELDS
    }
    if any(not isinstance(value, str) or not value.strip() or "<|" in value for value in values.values()):
        raise ValueError("Focus target requires a safe, nonempty question, entity, attribute and time condition")
    content = []
    for index in range(len(times)):
        content.extend(({"type": "text", "text": _frame_label(index, times)}, {"type": "image"}))
    request = (
        f"Original question\n{values['question']}\n\n"
        f"Entity\n{values['entity']}\n\n"
        f"Attribute or relation to verify\n{values['attribute']}\n\n"
        f"Time condition\n{values['time_condition']}\n\n"
        f"{INSTRUCTIONS}"
    )
    content.append({"type": "text", "text": request})
    return [{"role": "user", "content": content}]


def serialize_focus_report(raw, frame_times, *, capped=False):
    """Validate bounded integer frame citations (1..8) and render text for pure-text QA."""
    times = _checked_times(frame_times)
    if capped or not isinstance(raw, str):
        return "invalid", UNRESOLVED_REPORT
    response = raw.strip()
    if response == "UNRESOLVED":
        return "unresolved", UNRESOLVED_REPORT
    fenced = FENCE.fullmatch(response)
    try:
        parsed = json.loads(fenced[1] if fenced else response)
    except ValueError:
        return "invalid", UNRESOLVED_REPORT
    if not isinstance(parsed, dict) or set(parsed) != {"frames", "observation"}:
        return "invalid", UNRESOLVED_REPORT
    frames = parsed["frames"]
    if (
        not isinstance(frames, list)
        or not 1 <= len(frames) <= MAX_CITED_FRAMES
        or not all(type(index) is int and 0 <= index < len(times) for index in frames)
        or len(set(frames)) != len(frames)
    ):
        return "invalid", UNRESOLVED_REPORT
    observation = parsed["observation"]
    if not isinstance(observation, str):
        return "invalid", UNRESOLVED_REPORT
    observation = " ".join(observation.splitlines()).strip()
    if (
        not 1 <= len(observation) <= MAX_OBSERVATION_CHARS
        or "<|" in observation
        or ANSWER_LABEL.search(observation)
        or re.fullmatch(r"[a-zA-Z][.)]?", observation)
        or re.match(r"^[a-zA-Z][.)]\s", observation)
    ):
        return "invalid", UNRESOLVED_REPORT
    evidence = ", ".join(_frame_label(index, times) for index in sorted(frames))
    return "observed", f"Focus evidence from {evidence}. {observation}"
