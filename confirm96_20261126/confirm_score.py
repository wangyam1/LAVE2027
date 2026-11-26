"""Single-unsealing confirmation scorer for the 96 sealed CG-Bench long videos."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from missing_fact_routing_20261124.live_run import need
from missing_fact_routing_20261124.score import ANNOTATION_PIN, ARMS, paired

from .confirm_inputs import (
    ABI_BLOCK, ANNOTATION, N_CONFIRM, QID_ROLES_PIN, ROSTER_PIN, VAULT, load_scanner, pinned_bytes,
)
from .confirm_run import ACTION, PREFLIGHT, ROOT


ROUTES = (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))
ENDPOINTS = ("focus1_native", "focus12_on_policy", "focus12_equal_budget")


def _project_confirmation_gold_and_clues(cases, native_option, annotation_path, projection_path):
    """Single unsealing of right_answer & clue_intervals strictly for these 96 confirmation q1 rows."""
    roles = json.loads(pinned_bytes(VAULT / "qid_roles.private.json", QID_ROLES_PIN))
    need(roles["roster_sha256"] == ROSTER_PIN and roles["videos"] == N_CONFIRM, "Changed confirmation qid_roles")
    q1_by_uid = {r["video_uid"]: str(r["q1"]) for r in roles["rows"]}
    by_uid = {c["video_uid"]: c for c in cases}
    need(set(by_uid) == set(q1_by_uid) and len(by_uid) == N_CONFIRM, "Case UID set differs from sealed 96 roster")

    Projection = load_scanner(projection_path)
    scanner = Projection(pinned_bytes(annotation_path, ANNOTATION_PIN).decode("utf-8"))
    gold, clues = {}, {}
    for spans in scanner.records():
        uid = scanner.value(spans, "video_uid", {"video_uid"})
        if uid not in by_uid:
            continue
        qid = scanner.value(spans, "qid", {"qid"})
        if type(qid) not in (int, str) or str(qid) != q1_by_uid[uid]:
            continue
        case = by_uid[uid]
        question = scanner.value(spans, "question", {"question"})
        choices = scanner.value(spans, "choices", {"choices"})
        need(case["id"] not in gold and question == case["question"] and choices == case["choices"],
             f"Confirmation question/choices mismatch for {uid}")
        ans = scanner.value(spans, "right_answer", {"right_answer"})
        iv = scanner.value(spans, "clue_intervals", {"clue_intervals"}) or []
        need(type(ans) is str and native_option(ans, choices) == ans, f"Unexpected gold option for {uid}")
        gold[case["id"]] = ans
        clues[case["id"]] = iv
    need(len(gold) == len(clues) == N_CONFIRM
         and scanner.decoded["right_answer"] == N_CONFIRM
         and scanner.decoded["clue_intervals"] == N_CONFIRM,
         "Incomplete confirmation unsealing")
    return gold, clues


def _hits(indices, fps, intervals, exclude=()):
    if not intervals:
        return False
    ex = set(exclude)
    return any(any(lo <= (idx / fps) <= hi for lo, hi in intervals) for idx in indices if idx not in ex)


def _summarize_qa(items):
    correct = {
        arm: {ep: sum(ans[arm, ep] for ans, _ in items) for ep in ENDPOINTS}
        for arm in ARMS
    }
    routing = {
        ep: {
            f"{a}_vs_{b}": dict(
                paired(items, (a, ep), (b, ep)),
                distinct_f1_cases=sum(r[a] != r[b] for _, r in items),
            )
            for a, b in ROUTES
        }
        for ep in ENDPOINTS
    }
    f2_gain = {
        arm: {
            f"{ep}_vs_focus1": paired(items, (arm, ep), (arm, "focus1_native"))
            for ep in ("focus12_on_policy", "focus12_equal_budget")
        }
        for arm in ARMS
    }
    return dict(n=len(items), correct=correct, routing=routing, f2_gain=f2_gain)


def _summarize_clues(items):
    return dict(
        n=len(items),
        hits={arm: sum(h[arm] for h, _ in items) for arm in ARMS},
        pairs={f"{a}_vs_{b}": paired(items, a, b) for a, b in ROUTES},
    )


def score_confirmation(root, release_sha256, *, annotation_path, projection_path):
    root = Path(root)
    release = json.loads(pinned_bytes(root / "release.json", release_sha256))
    need(release["status"] == "confirm96_development_free_audited" and release["split"] == "confirmation"
         and release["gpu_released"] is True and release["cases"] == N_CONFIRM
         and release["labels"] == release["clues"] == release["training"] == 0,
         "Missing audited GPU-released confirmation run")
    terminal = json.loads(pinned_bytes(root / "action" / "terminal.json", release["terminal"]))
    manifest = json.loads(pinned_bytes(root / "preflight.json", terminal["pin"]))
    need(terminal["status"] == "complete" and terminal["cases"] == N_CONFIRM
         and manifest["order"] == terminal["order"], "Confirmation terminal/manifest mismatch")

    import sys
    sys.path.insert(0, str(ABI_BLOCK))
    import run_itt_utility as native

    cases = manifest["cases"]
    gold, clues = _project_confirmation_gold_and_clues(cases, native.option, annotation_path, projection_path)

    qa_strata = {name: [] for name in ("all", "legal", "f1_triggered", "f2_triggered")}
    f1_clue_strata = {name: [] for name in ("all", "legal", "f1_triggered")}
    f2_clue_counts = {arm: Counter() for arm in ARMS}
    k_dist = Counter()
    control_stats = Counter()
    timing_sums = Counter()
    timing_n = 0

    for ordinal, case in enumerate(cases):
        cid = case["id"]
        rec = json.loads(pinned_bytes(root / "action" / f"case_{ordinal:02d}" / "case.json", terminal["rows"][cid]))
        g = gold[cid]
        ivs = clues[cid]
        media = manifest["media"][cid]
        fps = media["fps"]
        src_set = set(media["overview_indices"])

        legal = rec["text_action"] is not None
        trace = rec.get("route_audit")
        addrs = (trace or {}).get("addresses") or []
        k_dist[len(addrs)] += 1
        if trace is not None:
            control_stats["scanned"] += 1
            if trace.get("decision") is None:
                control_stats["no_nomination"] += 1
            elif addrs:
                control_stats["approved"] += 1
            else:
                control_stats["rejected_by_controls"] += 1

        routes_f1 = {a: tuple(rec["routes"]["indices"][a]) for a in ARMS}
        f1_triggered = len(set(routes_f1.values())) > 1
        f2_triggered = bool(rec["focus2_plan"]["triggered"])

        ans_map = {}
        f1_hit_map = {}
        for arm in ARMS:
            a1 = (rec["qa1"][arm]["qa"] or {}).get("answer")
            a2_op = rec["qa2"][arm]["on_policy_answer"]
            a2_eq = rec["qa2"][arm]["equal_budget_answer"]
            ans_map[arm, "focus1_native"] = (a1 == g)
            ans_map[arm, "focus12_on_policy"] = (a2_op == g)
            ans_map[arm, "focus12_equal_budget"] = (a2_eq == g)
            f1_hit_map[arm] = _hits(rec["routes"]["indices"][arm], fps, ivs, exclude=src_set)
            f2_idx = rec["focus2_plan"]["arms"][arm]["indices"]
            kind = rec["focus2_plan"]["arms"][arm]["kind"]
            f2_clue_counts[arm][f"kind_{kind}"] += 1
            if f2_idx:
                h2 = _hits(f2_idx, fps, ivs)
                f2_clue_counts[arm][f"{kind}_clue_hits"] += int(h2)
                f2_clue_counts[arm]["total_clue_hits"] += int(h2)

        qa_item = (ans_map, routes_f1)
        clue_item = (f1_hit_map, routes_f1)
        qa_strata["all"].append(qa_item)
        f1_clue_strata["all"].append(clue_item)
        if legal:
            qa_strata["legal"].append(qa_item)
            f1_clue_strata["legal"].append(clue_item)
        if f1_triggered:
            qa_strata["f1_triggered"].append(qa_item)
            f1_clue_strata["f1_triggered"].append(clue_item)
        if f2_triggered:
            qa_strata["f2_triggered"].append(qa_item)

        for k_t, v_t in rec.get("timings", {}).items():
            timing_sums[k_t] += float(v_t)
        timing_sums["elapsed_seconds"] += float(rec.get("elapsed_seconds", 0.0))
        timing_n += 1

    return dict(
        status="confirm96_scored",
        split="confirmation",
        release_sha256=release_sha256,
        confirmation_labels=N_CONFIRM,
        clue_labels=N_CONFIRM,
        development_labels=0,
        gpu=0,
        k_distribution=dict(sorted(k_dist.items())),
        control_stats=dict(control_stats),
        mean_stage_seconds={k: round(v / timing_n, 2) for k, v in sorted(timing_sums.items())},
        qa={name: _summarize_qa(items) for name, items in qa_strata.items()},
        focus1_new_clue_hits={name: _summarize_clues(items) for name, items in f1_clue_strata.items()},
        focus2_clue_hits={a: dict(c) for a, c in f2_clue_counts.items()},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_sha256")
    parser.add_argument("annotation", type=Path, nargs="?", default=ANNOTATION)
    parser.add_argument("projection", type=Path, nargs="?", default=ROOT / "scoring" / "metadata_projection_scanner.py")
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Confirmation scoring requires hidden GPUs")
    output = score_confirmation(ROOT, args.release_sha256,
                                annotation_path=args.annotation, projection_path=args.projection)
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
