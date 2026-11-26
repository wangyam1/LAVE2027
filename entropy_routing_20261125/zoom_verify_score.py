"""Score the audited Focus-2 dense verification experiment on the 51 exposed dev videos."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from missing_fact_routing_20261124.score import (
    ARMS, N, closed, need, paired, pinned, project_labels,
)

from .score import _dev_clues
from .zoom_verify_run import MODES, PRIOR, PRIOR_RELEASE, PRIOR_SCORE, ROOT


ROUTES = (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))
POLICIES = ("baseline", "on_policy", "equal_budget")


def _summarize_stratum(items):
    correct = {
        arm: {
            "baseline": sum(ans[arm, "baseline"] for ans, _ in items),
            **{
                f"{mode}_{pol}": sum(ans[arm, f"{mode}_{pol}"] for ans, _ in items)
                for mode in MODES for pol in ("on_policy", "equal_budget")
            },
        }
        for arm in ARMS
    }
    routing = {}
    for key in ("baseline", *(f"{m}_{p}" for m in MODES for p in ("on_policy", "equal_budget"))):
        routing[key] = {
            f"{a}_vs_{b}": paired(items, (a, key), (b, key))
            for a, b in ROUTES
        }
    gain_vs_baseline = {
        arm: {
            f"{m}_{p}": paired(items, (arm, f"{m}_{p}"), (arm, "baseline"))
            for m in MODES for p in ("on_policy", "equal_budget")
        }
        for arm in ARMS
    }
    return dict(n=len(items), correct=correct, routing=routing, gain_vs_baseline=gain_vs_baseline)


def score_development(root, release_sha256, *, rows, native_option, annotation_path, qid_selection_path,
                      projection_path):
    root, rows = Path(root), list(rows)
    order = [row.cid for row in rows]
    release = json.loads(pinned(root / "release.json", release_sha256))
    need(release["status"] == "zoom_verify_development_audited" and release["gpu_released"] is True
         and release["cases"] == N and closed(release), "Missing completed Focus-2 GPU release audit")
    terminal = json.loads(pinned(root / "action" / "terminal.json", release["terminal"]))
    need(terminal["status"] == "complete" and terminal["cards"] == 1 and closed(terminal)
         and terminal["order"] == order and terminal["cases"] == N, "Wrong Focus-2 terminal")

    prior_score = json.loads(pinned(PRIOR / "score.json", PRIOR_SCORE))
    need(prior_score["status"] == "entropy_routing_development_scored"
         and prior_score["release_sha256"] == PRIOR_RELEASE, "Wrong parent entropy score")

    gold = project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path)
    clues = _dev_clues(rows, annotation_path)

    strata = {"all": [], "legal": [], "zoom_triggered": []}
    f2_clue_hits = {arm: Counter() for arm in ARMS}
    arm_kinds = {arm: Counter() for arm in ARMS}
    for ordinal, row in enumerate(rows):
        rec = json.loads(pinned(root / "action" / f"case_{ordinal:02d}" / "case.json", terminal["rows"][row.cid]))
        g = gold[row.cid]
        fps = row.media["fps"]
        ivs = clues[row.cid]
        legal = rec["plan"]["arms"]["text"]["kind"] != "common_invalid"
        ans_map = {}
        for arm in ARMS:
            cell = rec["cells"][arm]
            kind = cell["kind"]
            arm_kinds[arm][kind] += 1
            ans_map[arm, "baseline"] = (cell["baseline_answer"] == g)
            for mode in MODES:
                m = cell["modes"][mode]
                ans_map[arm, f"{mode}_on_policy"] = (m["on_policy_answer"] == g)
                ans_map[arm, f"{mode}_equal_budget"] = (m["equal_budget_answer"] == g)
            indices = rec["plan"]["arms"][arm]["indices"]
            if indices:
                hit = any(any(lo <= f / fps <= hi for lo, hi in ivs) for f in indices)
                f2_clue_hits[arm][f"{kind}_calls"] += 1
                f2_clue_hits[arm][f"{kind}_clue_hits"] += int(hit)
                f2_clue_hits[arm]["total_clue_hits"] += int(hit)
        item = (ans_map, rec["plan"]["triggered"])
        strata["all"].append(item)
        if legal:
            strata["legal"].append(item)
        if rec["plan"]["triggered"]:
            strata["zoom_triggered"].append(item)

    summary = {name: _summarize_stratum(items) for name, items in strata.items()}
    # Verify exact reproduction of parent Focus-1 native baseline totals
    for arm in ARMS:
        need(summary["all"]["correct"][arm]["baseline"] == prior_score["strata"]["all"]["correct"][arm]["native"],
             "Parent Focus-1 baseline QA did not reproduce")

    return dict(
        status="zoom_verify_development_scored", release_sha256=release_sha256,
        parent_release_sha256=PRIOR_RELEASE, parent_score_sha256=PRIOR_SCORE,
        development_labels=N, confirmation_labels=0, clue_labels=N, gpu=0,
        arm_kinds={a: dict(c) for a, c in arm_kinds.items()},
        f2_clue_hits={a: dict(c) for a, c in f2_clue_hits.items()},
        strata=summary,
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
