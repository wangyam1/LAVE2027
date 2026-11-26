"""Strictly blind loader for the 96 sealed CG-Bench long confirmation videos."""

from hashlib import sha256
import json
from pathlib import Path
from unicodedata import normalize

from missing_fact_routing_20261124.live_run import need
from missing_fact_routing_20261124.score import ANNOTATION_PIN, SCANNER_PIN
from missing_fact_routing_20261124.target import ATTRIBUTES, target_from_question


VAULT = Path("/data1/llj/lljmas/qwen35_cgbench_long_confirm_assets_20261121_01")
ANNOTATION = Path("/data1/llj/development/lljvideo/CLAUDEvidoe/CG-Bench/cgbench.json")
ROSTER_PIN = "3c7d22e0e461529a8aa699f4b3c0921677fcf008c5c80bb7f6d1f0214749553a"
QID_ROLES_PIN = "8179e40ca3dc3fab7aaab63bbfa73b03063380ec638f412770d998d019142ce0"
INTEGRITY_PIN = "87ed33eb745f2475201f33c52162c0f68d2eb0f30260ef1d1f708d1906ef4938"
ABI_BLOCK = Path("/data1/llj/lljmas/qwen35_native_itt_fresh96_20261116_04/blocks/block_02")
N_CONFIRM = 96


def pinned_bytes(path, pin):
    raw = Path(path).read_bytes()
    need(sha256(raw).hexdigest() == pin, f"Seal mismatch: {Path(path).name}")
    return raw


def hash_file(path):
    h = sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_scanner(projection_path):
    ns = {}
    exec(compile(pinned_bytes(projection_path, SCANNER_PIN), "frozen_metadata_projection", "exec"), ns)
    return ns["Projection"]


def load_blind_confirmation_cases(projection_path, *, annotation_path=ANNOTATION, vault=VAULT, dev_uids=()):
    """Load the 96 confirmation public cases decoding ONLY question/choices/duration (0 labels, 0 clues)."""
    roster = json.loads(pinned_bytes(vault / "roster.private.json", ROSTER_PIN))
    roles = json.loads(pinned_bytes(vault / "qid_roles.private.json", QID_ROLES_PIN))
    integrity = json.loads(pinned_bytes(vault / "receipts" / "first20261121" / "integrity.private.json", INTEGRITY_PIN))

    need(roster["n"] == len(roster["rows"]) == N_CONFIRM and roles["videos"] == len(roles["rows"]) == N_CONFIRM
         and len(integrity) == N_CONFIRM and roles["roster_sha256"] == ROSTER_PIN
         and roles["annotation_sha256"] == ANNOTATION_PIN and roles["scanner_sha256"] == SCANNER_PIN,
         "Sealed 96-video confirmation manifest mismatch")

    roster_by_uid = {r["member"]["video_uid"]: r for r in roster["rows"]}
    ordered_roles = sorted(roles["rows"], key=lambda r: r["rank"])
    need([r["rank"] for r in ordered_roles] == list(range(1, N_CONFIRM + 1)), "Invalid confirmation rank order")
    need(set(roster_by_uid) == {r["video_uid"] for r in ordered_roles} == set(integrity),
         "Confirmation UID sets disagree")
    if dev_uids:
        need(not (set(roster_by_uid) & set(dev_uids)), "Confirmation cohort overlaps exposed development videos!")

    q1_by_uid = {r["video_uid"]: str(r["q1"]) for r in ordered_roles}
    Projection = load_scanner(projection_path)
    scanner = Projection(pinned_bytes(annotation_path, ANNOTATION_PIN).decode("utf-8"))

    public_meta = {}
    for spans in scanner.records():
        uid = scanner.value(spans, "video_uid", {"video_uid"})
        if uid not in q1_by_uid:
            continue
        qid = scanner.value(spans, "qid", {"qid"})
        if type(qid) not in (int, str) or str(qid) != q1_by_uid[uid]:
            continue
        question = scanner.value(spans, "question", {"question"})
        choices = scanner.value(spans, "choices", {"choices"})
        duration = scanner.value(spans, "duration", {"duration"})
        need(uid not in public_meta and isinstance(question, str) and question.strip()
             and isinstance(choices, list) and 2 <= len(choices) <= 26,
             f"Invalid confirmation public metadata for {uid}")
        public_meta[uid] = dict(qid=str(qid), question=question, choices=choices, duration_annotated=duration)

    # Cryptographic blindness assertion: ZERO right_answer or clue_intervals decoded!
    need(len(public_meta) == N_CONFIRM
         and set(scanner.decoded) == {"video_uid", "qid", "question", "choices", "duration"}
         and all(scanner.decoded[k] == N_CONFIRM for k in ("question", "choices", "duration"))
         and scanner.decoded.get("right_answer", 0) == 0
         and scanner.decoded.get("clue_intervals", 0) == 0,
         "Blind projection violated or incomplete")

    cases = []
    for role in ordered_roles:
        uid = role["video_uid"]
        rank = role["rank"]
        meta = public_meta[uid]
        integ = integrity[uid]
        vpath = Path(integ["path"])
        need(vpath.parent == vault / "videos" and vpath.name == f"{uid}.mp4", "Unexpected confirmation MP4 path")
        cid = f"cgb_confirm_{rank:03d}"
        cases.append(dict(
            id=cid,
            rank=rank,
            seed_index=rank - 1,
            qid=meta["qid"],
            video_uid=uid,
            question=meta["question"],
            choices=meta["choices"],
            duration_annotated=float(meta["duration_annotated"]) if meta["duration_annotated"] is not None else float(roster_by_uid[uid]["min_duration_s"]),
            video_path=str(vpath),
            video_bytes=int(integ["bytes"]),
            video_sha256=integ["sha256"],
            split="confirmation",
        ))
    return cases


def abi_case(case):
    """Return exact frozen acquisition_contract dict without mutating confirmation provenance."""
    return dict(
        id=case["id"],
        video_uid=case["video_uid"],
        question=case["question"],
        choices=case["choices"],
        duration_annotated=case["duration_annotated"],
        video_path=case["video_path"],
        video_bytes=case["video_bytes"],
        split="development",
    )


def plan_confirm_foils(cases, tokenizer):
    """Exact same deterministic label-blind foil matcher as dev foil_plan.plan_foils."""
    recognized = {name for name, _ in ATTRIBUTES}
    profiles = {}
    for case in cases:
        need(case.get("split") == "confirmation", "Expected confirmation cases")
        cid, uid = case["id"], case["video_uid"]
        need(isinstance(cid, str) and cid and cid not in profiles and isinstance(uid, str) and uid.strip(),
             "Duplicate or invalid confirmation case ID")
        target = target_from_question(case["question"])
        fields = (target.question, target.entity, target.attribute, target.time_condition)
        count = len(tokenizer.encode("\n".join(fields), add_special_tokens=False))
        qkey = " ".join(normalize("NFKC", target.question).casefold().split()).rstrip(" ?")
        profiles[cid] = dict(video=uid, question=target.question, key=qkey,
                             attribute=target.attribute, tokens=count)
    need(len(profiles) == N_CONFIRM, "Expected 96 confirmation profiles for foils")
    plan = {}
    for cid, actual in sorted(profiles.items()):
        eligible = [
            other for other, foil in profiles.items()
            if foil["video"] != actual["video"] and foil["key"] != actual["key"]
        ]
        need(bool(eligible), f"No distinct foil question for {cid}")

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
