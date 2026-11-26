"""Render and parse an independent report from newly sampled Focus frames."""

from collections.abc import Mapping
import json
from math import isfinite

from .report import FENCE, FIELDS, INSTRUCTIONS, UNRESOLVED_REPORT, serialize_focus_report


def _view(frame_indices, fps):
    indices = list(frame_indices)
    if (not 1 <= len(indices) <= 32 or any(type(i) is not int or i < 0 for i in indices)
            or indices != sorted(set(indices)) or type(fps) not in (int, float)
            or not isfinite(fps) or fps <= 0):
        raise ValueError("Expected a chronological collection of distinct video frame indices and a positive frame rate")
    times = [i / fps for i in indices]
    if not all(map(isfinite, times)):
        raise ValueError("Refocus needs finite video timestamps")
    return indices, times


def render_refocus_prompt(target, frame_indices, fps):
    """Keep the native image slots and use the same text for matched views."""
    indices, _ = _view(frame_indices, fps)
    scope = {k: target.get(k) if isinstance(target, Mapping) else getattr(target, k, None) for k in FIELDS}
    if any(not isinstance(x, str) or not x.strip() or "<|" in x for x in scope.values()):
        raise ValueError("Refocus needs the original question and its evidence target")
    content = []
    for slot in range(len(indices)):
        content.extend((dict(type="text", text=f"Frame {slot}"), dict(type="image")))
    request = (
        "Read these sampled video frames in the order shown. Quoted text describes the requested evidence.\n\n"
        + "\n".join(f"{label}\n{json.dumps(scope[key], ensure_ascii=False)}" for key, label in (
            ("question", "Original question"),
            ("entity", "Entity"),
            ("attribute", "Attribute or relation to verify"),
            ("time_condition", "Time condition"),
        ))
        + f"\n\n{INSTRUCTIONS} Treat the cited frame numbers as local labels. "
        "If a count or duration requires parts of the video that are absent, report UNRESOLVED."
    )
    content.append(dict(type="text", text=request))
    return [dict(role="user", content=content)]


def _unique_fields(pairs):
    fields = dict(pairs)
    if len(fields) != len(pairs):
        raise ValueError("Duplicate report field")
    return fields


def parse_refocus_report(raw, frame_indices, fps, *, capped=False):
    """Use the native report format and retain exact local and video provenance."""
    indices, times = _view(frame_indices, fps)
    status, observation = serialize_focus_report(raw, times, capped=capped)
    parsed = dict(status=status, claim=None, citations=[], frame_indices=[], timestamps=[], observation=observation)
    if status != "observed":
        return parsed

    response = raw.strip()
    fenced = FENCE.fullmatch(response)
    try:
        report = json.loads(fenced[1] if fenced else response, object_pairs_hook=_unique_fields)
    except ValueError:
        return dict(parsed, status="invalid", observation=UNRESOLVED_REPORT)
    citations = sorted(report["frames"])
    return dict(parsed, claim=report["observation"].strip(), citations=citations,
                frame_indices=[indices[slot] for slot in citations], timestamps=[times[slot] for slot in citations])


def merge_refocus_report(original_observation, original_claim, parsed):
    """Use a new cited fact while preserving the original QA text on no change."""
    if (not isinstance(original_observation, str) or "<|" in original_observation
            or (original_claim is not None and not isinstance(original_claim, str))):
        raise ValueError("Expected the original native QA observation and optional original claim")
    status = parsed["status"]
    if status in ("unresolved", "invalid"):
        return dict(observation=original_observation, origin="original", relation="no_new_statement", status=status)
    if status != "observed" or not parsed["claim"] or not parsed["citations"]:
        raise ValueError("Expected a parsed Refocus report")
    if original_claim is not None and parsed["claim"] == original_claim.strip():
        return dict(observation=original_observation, origin="original", relation="same_literal_statement", status=status)
    # Different strings alone cannot establish entailment or contradiction.
    relation = "unassessed" if original_claim else "new_statement"
    return dict(observation=parsed["observation"], origin="refocus", relation=relation, status=status)
