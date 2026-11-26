"""Score the audited entropy-routing 3x2 factorial on the 51 exposed development videos."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from missing_fact_routing_20261124.score import (
    ANNOTATION_PIN, ARMS, MODES, N, case_outcomes, closed, need, paired, pinned, project_labels, summarize,
)

from .live_run import ROOT


def _dev_clues(rows, annotation_path):
    """Extract official clue_intervals strictly for these 51 exposed development rows."""
    ann = json.loads(pinned(annotation_path, ANNOTATION_PIN).decode("utf-8"))
    by_vq = {(item.get("video_uid"), item.get("question")): item.get("clue_intervals") or [] for item in ann}
    clues = {}
    for row in rows:
        key = (row.case["video_uid"], row.case["question"])
        need(key in by_vq, "Exposed development row missing from pinned annotation")
        clues[row.cid] = by_vq[key]
    need(len(clues) == N, "Expected exactly 51 exposed development clue records")
    return clues


def _new_clue_hit(indices, source_set, fps, intervals):
    if not intervals:
        return False
    return any(any(lo <= (idx / fps) <= hi for lo, hi in intervals) for idx in indices if idx not in source_set)


def score_development(root, release_sha256, *, rows, native_option, annotation_path, qid_selection_path,
                      projection_path):
    """Verify completed release audit, then score QA and official dev clue hits on 51 exposed videos only."""
    root, rows = Path(root), list(rows)
    release = json.loads(pinned(root / "release.json", release_sha256))
    need(release["status"] == "entropy_routing_factorial_audited" and release["gpu_released"] is True
         and release["cases"] == N and closed(release), "Missing completed label-free GPU-release audit")
    terminal = json.loads(pinned(root / "action" / "terminal.json", release["terminal"]))
    cids = [row.cid for row in rows]
    uids = {row.case["video_uid"] for row in rows}
    need(terminal["status"] == "complete" and terminal["cards"] == 1 and closed(terminal)
         and terminal["cases"] == len(cids) == len(set(cids)) == len(uids) == N
         and terminal["order"] == cids and set(terminal["rows"]) == set(cids)
         and all(row.case["split"] == "development" for row in rows), "Wrong audited development roster")

    answers, routing_meta = {}, {}
    flags = {arm: {mode: Counter() for mode in MODES} for arm in ARMS}
    counted = Counter()
    k_counts = Counter()
    for ordinal, row in enumerate(rows):
        work = root / "action" / f"case_{ordinal:02d}"
        record = json.loads(pinned(work / "case.json", terminal["rows"][row.cid]))
        outcome, ids, legal = case_outcomes(work, row, ordinal, record, native_option)
        answers[row.cid] = (outcome, ids, legal)
        trace = record.get("route_audit") or {}
        addrs = trace.get("addresses") or []
        routing_meta[row.cid] = dict(k=len(addrs), approved=bool(addrs),
                                     fps=row.media["fps"], source_set=set(row.media["overview_indices"]))
        k_counts[len(addrs)] += 1
        counted["qa"] += record["unique_qa_calls"]
        counted["invalid_action"] += not legal
        for (arm, mode), (_, report, qa, capped) in outcome.items():
            counts = flags[arm][mode]
            counts[f"report_{report}"] += 1
            counts[f"qa_{qa}"] += 1
            counts["report_capped"] += capped
            counted["invalid_qa_cells"] += qa in ("invalid_option", "capped")
    need(all(counted[key] == release[key] for key in ("qa", "invalid_action", "invalid_qa_cells"))
         and counted["qa"] == terminal["counts"]["unique_qa"]
         and counted["invalid_action"] == terminal["counts"]["invalid_action"], "Changed audited scoring totals")

    gold = project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path)
    clues = _dev_clues(rows, annotation_path)

    strata = {"all": [], "legal": [], "triggered": []}
    clue_strata = {"all": [], "legal": [], "triggered": []}
    for cid, (outcome, routes, legal) in answers.items():
        comparison = ({cell: result[0] == gold[cid] for cell, result in outcome.items()}, routes)
        meta = routing_meta[cid]
        clue_hit = {arm: _new_clue_hit(routes[arm], meta["source_set"], meta["fps"], clues[cid]) for arm in ARMS}
        clue_item = (clue_hit, routes)
        strata["all"].append(comparison)
        clue_strata["all"].append(clue_item)
        if legal:
            strata["legal"].append(comparison)
            clue_strata["legal"].append(clue_item)
        if len(set(routes.values())) > 1:
            strata["triggered"].append(comparison)
            clue_strata["triggered"].append(clue_item)

    clue_summary = {}
    for name, items in clue_strata.items():
        clue_summary[name] = dict(
            n=len(items),
            new_hits={arm: sum(hit[arm] for hit, _ in items) for arm in ARMS},
            pairs={
                f"{a}_vs_{b}": paired(items, a, b)
                for a, b in (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))
            },
        )

    return dict(
        status="entropy_routing_development_scored", release_sha256=release_sha256,
        development_labels=N, confirmation_labels=0, clue_labels=N, gpu=0,
        qa_calls=counted["qa"], invalid_actions=counted["invalid_action"],
        k_distribution=dict(sorted(k_counts.items())),
        cells=flags,
        strata={name: summarize(results) for name, results in strata.items()},
        dev_new_clue_hits=clue_summary,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_sha256")
    for name in ("annotation", "qid_selection", "projection"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Development scoring requires hidden GPUs")
    from missing_fact_routing_20261124.live_run import load_history

    _, history, _ = load_history()
    import run_itt_utility as native

    output = score_development(ROOT, args.release_sha256, rows=history.rows, native_option=native.option,
                               annotation_path=args.annotation, qid_selection_path=args.qid_selection,
                               projection_path=args.projection)
    need(not history.s.torch.cuda.is_initialized(), "Scoring initialized CUDA")
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
