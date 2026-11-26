"""Score the audited follow-up Focus on the original exposed development cohort."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path

from .citation_score import COMPARISONS, DECISIONS, PRIOR, PRIOR_RELEASE, PRIOR_SCORE, _terminal
from .score import ARMS, N, case_outcomes, need, paired, pinned, project_labels


ROOT = Path(__file__).resolve().parents[1]
PRIOR_CITATION_RELEASE = "731ff8b83014f85bd4ac2140994ce4eaeeab8bf52411f72899e1687b59909556"
ROUTES = ("local", "clock")
MODES = ("baseline", "local", "clock", "local_verified", "clock_verified")
CELL_FIELDS = {*MODES, "planned"}
PAIRS = (
    ("local", "baseline"), ("clock", "baseline"), ("local", "clock"),
    ("local_verified", "baseline"), ("clock_verified", "baseline"),
    ("local_verified", "clock_verified"),
    ("local_verified", "local"), ("clock_verified", "clock"),
)
PLAN_STATES = ("bypass", "no_plan", "planned")


def _readout(read, baseline, counts):
    state = read["measurement"]["report"]["status"]
    origin = read["merge"]["origin"]
    qa, verifier = read["qa"], read["verification"]
    need(state in ("invalid", "unresolved", "observed") and origin in ("original", "refocus"),
         "Unexpected follow-up report or merge status")
    counts.update(physical_reads=1, **{f"report_{state}": 1})
    if origin == "original":
        need(qa is None and verifier is None, "A fallback generated follow-up QA")
        return baseline, baseline
    need(state == "observed" and qa is not None and verifier is not None, "An edited report lacks matching QA and verification")
    decision = verifier["decisions"]["cited"]
    need(decision in DECISIONS, "Unexpected follow-up verification decision")
    counts.update(qa=1, verified=int(decision == "SUPPORT"), **{f"cited_{decision}": 1})
    return qa["answer"], qa["answer"] if decision == "SUPPORT" else baseline


def _case(row, ordinal, original, baseline, record, native_option):
    need(record["cid"] == row.cid and record["ordinal"] == ordinal and record["case_uid"] == row.case["video_uid"]
         and set(record["cells"]) == set(ARMS) and isinstance(record["claims"], dict),
         "Follow-up results changed the original case or arms")
    cells, claims = record["cells"], record["claims"]
    targeted = original["focus"]["targeted"]
    focus = targeted["rows"]
    owners = {
        arm: focus[arm].get("reused_identical_input", arm)
        for arm in ARMS if baseline[arm, "targeted"][1] == "observed"
    }
    need(set(claims) <= set(owners.values()), "Follow-up used an unobserved or nonphysical prior claim")
    plans, by_origin = {}, {}
    readouts = {route: Counter() for route in ROUTES}
    for origin, claim in claims.items():
        need(focus[origin]["calls"] == 1 and isinstance(claim, dict), "Invalid physical claim origin")
        need(claim["arms"] == [arm for arm in targeted["execution_order"] if owners.get(arm) == origin],
             "Follow-up claim changed its logical aliases")
        plan = claim["plan"]
        need(plan["status"] in ("no_plan", "planned"), "Unexpected follow-up plan status")
        if plan["status"] == "planned":
            need(plan["reason"] is None and set(claim["reads"]) == set(ROUTES), "Incomplete planned follow-up pair")
            by_origin[origin] = {
                route: _readout(claim["reads"][route], baseline[origin, "targeted"][0], readouts[route])
                for route in ROUTES
            }
        else:
            need(isinstance(plan["reason"], str) and plan["reason"] and not claim["reads"],
                 "Fallback lacks a plan reason or acquired unexpected frames")
        plans[origin] = plan

    states = {}
    for arm in ARMS:
        cell = cells[arm]
        owner = owners.get(arm)
        status = plans.get(owner, {}).get("status", "bypass")
        need(set(cell) == CELL_FIELDS and type(cell["planned"]) is bool and cell["planned"] == (status == "planned")
             and cell["baseline"] == baseline[arm, "targeted"][0], "Follow-up changed its plan or original answer")
        need(all(cell[mode] is None or (type(cell[mode]) is str
                 and native_option(cell[mode], row.case["choices"]) == cell[mode]) for mode in MODES),
             "Follow-up answer is not a native option")
        if not cell["planned"]:
            need(all(cell[mode] == cell["baseline"] for mode in MODES), "A bypassed plan changed the prior answer")
        else:
            need(all((cell[route], cell[f"{route}_verified"]) == by_origin[owner][route] for route in ROUTES),
                 "Follow-up answer differs from its audited read or verification")
        states[arm] = status
    for arm, owner in owners.items():
        need(cells[arm] == cells[owner], "A reused physical claim has different follow-up answers")
    return cells, states, plans, readouts


def _endpoint(items):
    pairs = {f"{after}_vs_{before}": paired(items, after, before) for after, before in PAIRS}
    return dict(n=len(items), correct={mode: sum(answer[mode] for answer, _ in items) for mode in MODES}, pairs=pairs)


def _summary(items):
    return dict(
        n=len(items),
        by_arm={arm: _endpoint([({mode: answer[arm, mode] for mode in MODES}, None) for answer, _ in items])
                for arm in ARMS},
        baseline_routing={
            f"{a}_vs_{b}": dict(paired(items, (a, "baseline"), (b, "baseline")),
                                 distinct_frame_cases=sum(route[a] != route[b] for _, route in items))
            for a, b in COMPARISONS
        },
    )


def score_development(root, release_sha256, *, prior_root, prior_release_sha256, prior_score_sha256,
                      rows, native_option, annotation_path, qid_selection_path, projection_path):
    """Verify the complete label-free execution before opening exposed answer keys."""
    root, prior_root, rows = Path(root), Path(prior_root), list(rows)
    order = [row.cid for row in rows]
    need(len(order) == len(set(order)) == len({row.case["video_uid"] for row in rows}) == N
         and all(row.case["split"] == "development" and row.case["id"] == row.cid for row in rows),
         "Invalid exposed development scope")
    terminal = _terminal(root, release_sha256, "refocus_development_audited", order)
    release = json.loads(pinned(root / "release.json", release_sha256))
    preflight = json.loads(pinned(root / "preflight.json", terminal["pin"]))
    need(preflight["status"] == "refocus_inputs_frozen" and preflight["parent"] == PRIOR_CITATION_RELEASE
         and preflight["order"] == order and preflight["labels"] == preflight["holdout"] == preflight["gpu"] == 0,
         "Changed follow-up or citation preflight")
    prior_terminal = _terminal(prior_root, prior_release_sha256, "missing_fact_factorial_audited", order)
    prior_score = json.loads(pinned(prior_root / "score.json", prior_score_sha256))
    need(prior_score["status"] == "missing_fact_development_scored"
         and prior_score["release_sha256"] == prior_release_sha256 and prior_score["development_labels"] == N
         and all(prior_score[k] == 0 for k in ("confirmation_labels", "clue_labels", "gpu")),
         "Wrong prior targeted result")

    results, plans_by_case, physical_states, reasons = {}, {}, Counter(), Counter()
    readouts = {route: Counter() for route in ROUTES}
    for ordinal, row in enumerate(rows):
        folder = f"case_{ordinal:02d}"
        prior_work = prior_root / "action" / folder
        original = json.loads(pinned(prior_work / "case.json", prior_terminal["rows"][row.cid]))
        baseline, routes, legal = case_outcomes(prior_work, row, ordinal, original, native_option)
        record = json.loads(pinned(root / "action" / folder / "case.json", terminal["rows"][row.cid]))
        cells, states, plans, counts = _case(row, ordinal, original, baseline, record, native_option)
        results[row.cid] = cells, states, routes, legal
        plans_by_case[row.cid] = plans
        physical_states.update(plan["status"] for plan in plans.values())
        reasons.update(plan["reason"] for plan in plans.values() if plan["reason"] is not None)
        for route in ROUTES:
            readouts[route].update(counts[route])

    plan_bytes = json.dumps(plans_by_case, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    totals = {key: sum(cnt[key] for cnt in readouts.values()) for key in ("physical_reads", "qa", "verified")}
    need(preflight["roster"] == sha256(plan_bytes).hexdigest()
         and preflight["arms"] == {cid: sorted(plans) for cid, plans in plans_by_case.items()}
         and preflight["claims"] == physical_states["planned"] and release["reads"] == totals["physical_reads"]
         and all(totals[key] == terminal["counts"][key] for key in totals), "Changed follow-up plan or work accounting")

    gold = project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path)
    strata = {name: [] for name in ("all", "legal", "triggered", "planned", "unplanned")}
    by_arm_plan = {arm: {state: [] for state in PLAN_STATES} for arm in ARMS}
    changes = {arm: {mode: Counter() for mode in MODES} for arm in ARMS}
    for cid, (cells, states, routes, legal) in results.items():
        correct = {(arm, mode): cells[arm][mode] == gold[cid] for arm in ARMS for mode in MODES}
        for arm, cell in cells.items():
            by_arm_plan[arm][states[arm]].append(({mode: correct[arm, mode] for mode in MODES}, None))
            for mode in MODES:
                changes[arm][mode].update(option_changed=cell[mode] != cell["baseline"], missing=cell[mode] is None)
        item = correct, routes
        strata["all"].append(item)
        if legal:
            strata["legal"].append(item)
        if len(set(routes.values())) > 1:
            strata["triggered"].append(item)
        strata["planned" if "planned" in states.values() else "unplanned"].append(item)

    summary = {name: _summary(items) for name, items in strata.items()}
    for name in ("all", "legal", "triggered"):
        scope, old = summary[name], prior_score["strata"][name]
        need(scope["n"] == old["n"] and scope["baseline_routing"] == old["routing"]["targeted"]
             and all(scope["by_arm"][arm]["correct"]["baseline"] == old["correct"][arm]["targeted"] for arm in ARMS),
             "Prior targeted totals or paired comparisons did not reproduce")
    return dict(
        status="refocus_development_scored", release_sha256=release_sha256,
        prior_release_sha256=prior_release_sha256, prior_score_sha256=prior_score_sha256,
        citation_release_sha256=PRIOR_CITATION_RELEASE,
        development_labels=N, confirmation_labels=0, clue_labels=0, gpu=0,
        strata=summary, changes=changes,
        by_arm_plan={arm: {state: _endpoint(items) for state, items in groups.items()}
                     for arm, groups in by_arm_plan.items()},
        physical_plans=dict(total=sum(physical_states.values()), counts=dict(physical_states),
                            no_plan_reasons=dict(reasons), readouts=readouts),
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
