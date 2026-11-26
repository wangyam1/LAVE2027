"""Score the audited F1-state choice of F2 time windows on 51 development videos."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .citation_score import COMPARISONS, PRIOR, PRIOR_RELEASE, PRIOR_SCORE, _terminal
from .refocus_score import PRIOR_CITATION_RELEASE, _readout
from .score import ARMS, N, case_outcomes, need, paired, pinned, project_labels


ROOT = Path(__file__).resolve().parents[1]
SIDES = ("before", "after")
PHYSICAL_ROUTES = dict(before="local", after="clock")
POLICIES = ("latent", "text", "clock")
MODES = ("baseline", *SIDES, *POLICIES, *(f"{name}_verified" for name in POLICIES))
CELL_FIELDS = {*MODES, "evaluated"}
PAIRS = (
    ("latent", "text"), ("latent", "clock"), ("text", "clock"),
    ("latent_verified", "text_verified"), ("latent_verified", "clock_verified"),
    ("text_verified", "clock_verified"),
    *((name, "baseline") for name in MODES if name != "baseline"),
)
ACTION_FIELDS = ("approved", "latent_differs_text", "latent_differs_clock", "text_differs_clock",
                 "approved_differs_clock")


def _decision(decision):
    selected = dict(latent=decision["state_side"], text=decision["text_side"], clock=decision["clock_side"])
    approved = decision["approved"]
    need(all(side in SIDES for side in selected.values()) and type(approved) is bool,
         "Follow-up decision has an invalid side or approval")
    need(approved or selected["latent"] == selected["text"], "An unapproved state changed the text action")
    flags = dict(approved=approved, latent_differs_text=selected["latent"] != selected["text"],
                 latent_differs_clock=selected["latent"] != selected["clock"],
                 text_differs_clock=selected["text"] != selected["clock"],
                 approved_differs_clock=approved and selected["latent"] != selected["clock"])
    return selected, flags


def _case(row, ordinal, original, baseline, record, native_option):
    need(record["cid"] == row.cid and record["ordinal"] == ordinal and record["case_uid"] == row.case["video_uid"]
         and set(record["cells"]) == set(ARMS), "Follow-up results changed the original case or arms")
    cells, decisions = record["cells"], record["decisions"]
    targeted = original["focus"]["targeted"]
    focus = targeted["rows"]
    owners = {
        arm: focus[arm].get("reused_identical_input", arm)
        for arm in ARMS if baseline[arm, "targeted"][1] == "observed"
    }
    need(set(decisions) == set(record["reads"]) <= set(owners.values()),
         "Follow-up decision lacks an original physical claim or matched reads")
    flags, expected = {}, {}
    physical = {side: Counter() for side in SIDES}
    side_roster = Counter()
    for origin, decision in decisions.items():
        need(focus[origin]["calls"] == 1 and decision["arms"] == [
            arm for arm in targeted["execution_order"] if owners.get(arm) == origin
        ], "Follow-up changed logical aliases of a physical claim")
        selected, flags[origin] = _decision(decision)
        side_roster.update(f"{policy}_{side}" for policy, side in selected.items())
        reads = record["reads"][origin]["reads"]
        need(set(reads) == set(PHYSICAL_ROUTES.values()), "Expected exactly two physical side reads")
        before = baseline[origin, "targeted"][0]
        outcomes = {side: _readout(reads[route], before, physical[side])
                    for side, route in PHYSICAL_ROUTES.items()}
        answer = {side: outcomes[side][0] for side in SIDES}
        for policy, side in selected.items():
            answer[policy], answer[f"{policy}_verified"] = outcomes[side]
        expected[origin] = answer

    flags_by_arm = {}
    for arm, cell in cells.items():
        owner = owners.get(arm)
        evaluated = owner in decisions
        base = baseline[arm, "targeted"][0]
        need(set(cell) == CELL_FIELDS and cell["evaluated"] is evaluated and cell["baseline"] == base,
             "Follow-up changed its evaluation scope or pinned baseline")
        need(all(cell[mode] is None or (type(cell[mode]) is str
                 and native_option(cell[mode], row.case["choices"]) == cell[mode]) for mode in MODES),
             "Follow-up answer is not a native option")
        if evaluated:
            need(all(cell[mode] == answer for mode, answer in expected[owner].items()),
                 "Policy answer differs from its selected physical read")
            flags_by_arm[arm] = flags[owner]
        else:
            need(all(cell[mode] == base for mode in MODES), "An unevaluated cell changed the baseline answer")
    return cells, flags_by_arm, flags, physical, side_roster


def _counts(items):
    return dict(evaluated=len(items), **{field: sum(value[field] for value in items) for field in ACTION_FIELDS})


def _endpoint(items):
    return dict(n=len(items), correct={mode: sum(outcome[mode] for outcome, _ in items) for mode in MODES},
                pairs={f"{a}_vs_{b}": paired(items, a, b) for a, b in PAIRS})


def _baseline_routing(items):
    return {
        f"{a}_vs_{b}": dict(paired(items, (a, "baseline"), (b, "baseline")),
                             distinct_frame_cases=sum(route[a] != route[b] for _, route in items))
        for a, b in COMPARISONS
    }


def score_development(root, release_sha256, *, prior_root, prior_release_sha256, prior_score_sha256,
                      rows, native_option, annotation_path, qid_selection_path, projection_path):
    """Check both released runs before opening the same 51 public development answers."""
    root, prior_root, rows = Path(root), Path(prior_root), list(rows)
    order = [row.cid for row in rows]
    need(len(order) == len(set(order)) == len({row.case["video_uid"] for row in rows}) == N
         and all(row.case["split"] == "development" and row.case["id"] == row.cid for row in rows),
         "Invalid exposed development scope")
    terminal = _terminal(root, release_sha256, "focus_side_development_audited", order)
    prior_terminal = _terminal(prior_root, prior_release_sha256, "missing_fact_factorial_audited", order)
    prior_score = json.loads(pinned(prior_root / "score.json", prior_score_sha256))
    need(prior_score["status"] == "missing_fact_development_scored"
         and prior_score["release_sha256"] == prior_release_sha256 and prior_score["development_labels"] == N
         and all(prior_score[k] == 0 for k in ("confirmation_labels", "clue_labels", "gpu")),
         "Wrong prior targeted result")

    results, physical_flags = {}, []
    physical_reads = {side: Counter() for side in SIDES}
    side_roster = Counter()
    for ordinal, row in enumerate(rows):
        folder = f"case_{ordinal:02d}"
        prior_work = prior_root / "action" / folder
        original = json.loads(pinned(prior_work / "case.json", prior_terminal["rows"][row.cid]))
        baseline, routes, legal = case_outcomes(prior_work, row, ordinal, original, native_option)
        record = json.loads(pinned(root / "action" / folder / "case.json", terminal["rows"][row.cid]))
        cells, flags_by_arm, flags, reads, labels = _case(row, ordinal, original, baseline, record, native_option)
        results[row.cid] = cells, flags_by_arm, routes, legal
        physical_flags.extend(flags.values())
        side_roster.update(labels)
        for side in SIDES:
            physical_reads[side].update(reads[side])

    gold = project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path)
    strata = {name: [] for name in ("all", "legal", "triggered")}
    by_arm = {arm: {scope: [] for scope in ("all", "evaluated", *ACTION_FIELDS)} for arm in ARMS}
    actions_by_arm = {arm: [] for arm in ARMS}
    answer_changes = {arm: {mode: Counter() for mode in MODES} for arm in ARMS}
    for cid, (cells, flags, routes, legal) in results.items():
        correctness = {(arm, mode): cells[arm][mode] == gold[cid] for arm in ARMS for mode in MODES}
        item = correctness, routes
        strata["all"].append(item)
        if legal:
            strata["legal"].append(item)
        if len(set(routes.values())) > 1:
            strata["triggered"].append(item)
        for arm, cell in cells.items():
            pair = {mode: correctness[arm, mode] for mode in MODES}, None
            by_arm[arm]["all"].append(pair)
            if arm in flags:
                actions_by_arm[arm].append(flags[arm])
                by_arm[arm]["evaluated"].append(pair)
                for field in ACTION_FIELDS:
                    if flags[arm][field]:
                        by_arm[arm][field].append(pair)
            for mode in MODES:
                answer_changes[arm][mode].update(option_changed=cell[mode] != cell["baseline"],
                                                 missing=cell[mode] is None)

    baseline_replay = {scope: dict(
        n=len(items), routing=_baseline_routing(items),
        correct={arm: sum(outcome[arm, "baseline"] for outcome, _ in items) for arm in ARMS},
    ) for scope, items in strata.items()}
    for name, scope in baseline_replay.items():
        old = prior_score["strata"][name]
        need(scope["n"] == old["n"] and scope["routing"] == old["routing"]["targeted"]
             and all(scope["correct"][arm] == old["correct"][arm]["targeted"] for arm in ARMS),
             "Prior targeted totals or paired comparisons did not reproduce")
    return dict(
        status="focus_side_development_scored", release_sha256=release_sha256,
        prior_release_sha256=prior_release_sha256, prior_score_sha256=prior_score_sha256,
        citation_release_sha256=PRIOR_CITATION_RELEASE,
        development_labels=N, confirmation_labels=0, clue_labels=0, gpu=0,
        baseline_replay=baseline_replay,
        endpoints={arm: {scope: _endpoint(items) for scope, items in group.items()} for arm, group in by_arm.items()},
        actions=dict(physical=_counts(physical_flags), by_arm={arm: _counts(items) for arm, items in actions_by_arm.items()},
                     physical_side_counts=side_roster),
        physical_reads=physical_reads, answer_changes=answer_changes,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_sha256")
    for name in ("annotation", "qid_selection", "projection"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Development scoring requires hidden GPUs")
    from .live_run import load_history

    _, history, _ = load_history()
    import run_itt_utility as native

    output = score_development(ROOT, args.release_sha256, prior_root=PRIOR, prior_release_sha256=PRIOR_RELEASE,
                               prior_score_sha256=PRIOR_SCORE, rows=history.rows, native_option=native.option,
                               annotation_path=args.annotation, qid_selection_path=args.qid_selection,
                               projection_path=args.projection)
    need(not history.s.torch.cuda.is_initialized(), "Scoring initialized CUDA")
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
