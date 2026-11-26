"""Freeze wrong-question controls from public development metadata."""

from unicodedata import normalize

from .target import ATTRIBUTES, target_from_question


def plan_foils(rows, tokenizer):
    """Map each native development row ID to a wrong question and its match audit.

    Matching first prefers the same recognized attribute, then the closest token
    count for the four variable target fields, then the smallest case ID. A foil
    may be used for multiple cases. Only public row metadata is read.
    """
    recognized = {name for name, _ in ATTRIBUTES}
    profiles = {}
    for row in rows:
        case = row.case
        if case.get("split") != "development":
            raise ValueError("Wrong-question controls require development rows")
        cid, video_uid = row.cid, case["video_uid"]
        if not isinstance(cid, str) or not cid or cid != case["id"] or cid in profiles:
            raise ValueError("Expected distinct native development case IDs")
        if not isinstance(video_uid, str) or not video_uid.strip():
            raise ValueError("Expected a public video UID for every case")
        target = target_from_question(case["question"])
        fields = (target.question, target.entity, target.attribute, target.time_condition)
        count = len(tokenizer.encode("\n".join(fields), add_special_tokens=False))
        question_key = " ".join(normalize("NFKC", target.question).casefold().split()).rstrip(" ?")
        profiles[cid] = dict(video=video_uid, question=target.question, key=question_key,
                             attribute=target.attribute, tokens=count)

    if len(profiles) < 2:
        raise ValueError("A wrong-question control requires at least two development cases")

    plan = {}
    for cid, actual in sorted(profiles.items()):
        eligible = [
            other for other, foil in profiles.items()
            if foil["video"] != actual["video"] and foil["key"] != actual["key"]
        ]
        if not eligible:
            raise ValueError(f"No distinct question from a different video for {cid}")

        def rank(other):
            foil = profiles[other]
            mismatch = actual["attribute"] in recognized and actual["attribute"] != foil["attribute"]
            return mismatch, abs(actual["tokens"] - foil["tokens"]), other

        foil_id = min(eligible, key=rank)
        foil = profiles[foil_id]
        plan[cid] = dict(foil_id=foil_id, question=foil["question"],
                         attribute=actual["attribute"], foil_attribute=foil["attribute"],
                         tokens=actual["tokens"], foil_tokens=foil["tokens"])
    return plan
