"""Score citation-filtered native text QA on the completed exposed development cohort."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from .score import ARMS, N, case_outcomes, closed, need, paired, pinned, project_labels


BASE = Path("/data1/llj/lljmas")
ROOT = Path(__file__).resolve().parents[1]
PRIOR = BASE / "qwen35_cgbench_missing_fact_20261124_01"
PRIOR_RELEASE = "fdc459c25af96fae4d1a0b4f240f9b2c7c9212871605deb5914098a573bed33a"
PRIOR_SCORE = "f695cb4f443615c182ea037f7e9be924f9574770e02f4b2078553894690af28e"
MODES = ("targeted", "verified")
DECISIONS = ("SUPPORT", "REJECT", "UNCERTAIN", "INVALID")
GATES = ("retained", "rejected", "bypass")
CELL_FIELDS = {"original_answer", "answer", "gate", "cited_decision", "decoy_decision"}
COMPARISONS = (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))


def _terminal(root, release_pin, status, order):
    release = json.loads(pinned(root / "release.json", release_pin))
    need(release["status"] == status and release["gpu_released"] is True
         and release["cases"] == N and closed(release), "Missing completed, label-free GPU-release audit")
    terminal = json.loads(pinned(root / "action" / "terminal.json", release["terminal"]))
    need(terminal["status"] == "complete" and terminal["cards"] == 1 and closed(terminal)
         and terminal["cases"] == len(order) == N and terminal["order"] == order
         and set(terminal["rows"]) == set(order), "Wrong audited development roster")
    return terminal


def _case(ordinal, row, prior, prior_outcome, candidate, native_option):
    need(candidate["cid"] == row.cid and candidate["ordinal"] == ordinal
         and candidate["case_uid"] == row.case["video_uid"] and set(candidate["cells"]) == set(ARMS),
         "Citation results changed the original case or arms")
    claims, cells = candidate["claims"], candidate["cells"]
    focus = prior["focus"]["targeted"]["rows"]
    owners, rejected = set(), set()
    for arm, cell in cells.items():
        original, report_status, _, _ = prior_outcome[arm, "targeted"]
        need(set(cell) == CELL_FIELDS and cell["original_answer"] == original,
             "Claim filtering changed the original targeted answer")
        answer = cell["answer"]
        need(answer is None or (type(answer) is str and answer == native_option(answer, row.case["choices"])),
             "A citation-filtered QA answer is not a native option")
        if report_status != "observed":
            need(cell == dict(original_answer=original, answer=original, gate="bypass",
                              cited_decision=None, decoy_decision=None), "Uncheckable report must bypass the gate")
            continue
        owner = focus[arm].get("reused_identical_input", arm)
        need(owner in ARMS and focus[owner]["calls"] == 1 and owner in claims,
             "A parsed citation lacks its original physical claim")
        owners.add(owner)
        cited, decoy = claims[owner]["cited_decision"], claims[owner]["decoy_decision"]
        gate = "retained" if cited == "SUPPORT" else "rejected"
        need(cited in DECISIONS and (decoy is None or decoy in DECISIONS)
             and (cell["gate"], cell["cited_decision"], cell["decoy_decision"]) == (gate, cited, decoy),
             "Gate or decisions differ from the physical cited view")
        if gate == "rejected":
            rejected.add(answer)
        else:
            need(answer == original, "A retained claim changed its QA answer")
    need(set(claims) == owners and len(rejected) <= 1, "Unexpected claim or multiple common fallback answers")
    return cells, [(item["cited_decision"], item["decoy_decision"]) for item in claims.values()]


def _summarize(results):
    return dict(
        n=len(results),
        correct={arm: {mode: sum(answer[arm, mode] for answer, _ in results) for mode in MODES} for arm in ARMS},
        verified_vs_targeted={arm: paired(results, (arm, "verified"), (arm, "targeted")) for arm in ARMS},
        routing={
            mode: {
                f"{a}_vs_{b}": dict(paired(results, (a, mode), (b, mode)),
                                     distinct_frame_cases=sum(route[a] != route[b] for _, route in results))
                for a, b in COMPARISONS
            }
            for mode in MODES
        },
    )


def _claim_summary(claims):
    paired_claims = [(cited == "SUPPORT", decoy == "SUPPORT") for cited, decoy in claims if decoy is not None]
    win = sum(cited and not decoy for cited, decoy in paired_claims)
    loss = sum(decoy and not cited for cited, decoy in paired_claims)
    return dict(
        unique_claims=len(claims),
        cited={decision: sum(cited == decision for cited, _ in claims) for decision in DECISIONS},
        decoy={decision: sum(decoy == decision for _, decoy in claims) for decision in DECISIONS},
        decoy_unavailable=len(claims) - len(paired_claims),
        joint_counts=dict(sorted(Counter(f"{cited}/{decoy or 'unavailable'}" for cited, decoy in claims).items())),
        cited_vs_decoy_support=dict(n=len(paired_claims), win=win, loss=loss,
                                    tie=len(paired_claims) - win - loss, net=win - loss),
    )


def score_development(root, release_sha256, *, prior_root, prior_release_sha256, prior_score_sha256,
                      rows, native_option, annotation_path, qid_selection_path, projection_path):
    """Validate pinned prior and successor results before projecting exactly 51 official answer keys."""
    root, prior_root, rows = Path(root), Path(prior_root), list(rows)
    order = [row.cid for row in rows]
    need(len(order) == len(set(order)) == len({row.case["video_uid"] for row in rows}) == N
         and all(row.case["split"] == "development" for row in rows), "Invalid exposed development scope")
    terminal = _terminal(root, release_sha256, "citation_check_development_audited", order)
    prior_terminal = _terminal(prior_root, prior_release_sha256, "missing_fact_factorial_audited", order)
    prior_score = json.loads(pinned(prior_root / "score.json", prior_score_sha256))
    need(prior_score["status"] == "missing_fact_development_scored"
         and prior_score["release_sha256"] == prior_release_sha256 and prior_score["development_labels"] == N
         and all(prior_score[k] == 0 for k in ("confirmation_labels", "clue_labels", "gpu")),
         "Wrong prior targeted result")

    results, unique_claims = {}, []
    for ordinal, row in enumerate(rows):
        folder = f"case_{ordinal:02d}"
        prior_work = prior_root / "action" / folder
        prior = json.loads(pinned(prior_work / "case.json", prior_terminal["rows"][row.cid]))
        original, routes, legal = case_outcomes(prior_work, row, ordinal, prior, native_option)
        candidate = json.loads(pinned(root / "action" / folder / "case.json", terminal["rows"][row.cid]))
        cells, claims = _case(ordinal, row, prior, original, candidate, native_option)
        results[row.cid] = cells, routes, legal
        unique_claims.extend(claims)

    gold = project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path)
    strata = {name: [] for name in ("all", "legal", "triggered")}
    by_gate = {arm: {gate: [] for gate in GATES} for arm in ARMS}
    changes = {arm: Counter() for arm in ARMS}
    for cid, (cells, routes, legal) in results.items():
        correct = {}
        for arm, cell in cells.items():
            before, after = cell["original_answer"], cell["answer"]
            targeted, verified = before == gold[cid], after == gold[cid]
            correct[arm, "targeted"], correct[arm, "verified"] = targeted, verified
            by_gate[arm][cell["gate"]].append((dict(targeted=targeted, verified=verified), None))
            changes[arm].update(option_changed=before != after, targeted_missing=before is None,
                                verified_missing=after is None)
        pair = correct, routes
        strata["all"].append(pair)
        if legal:
            strata["legal"].append(pair)
        if len(set(routes.values())) > 1:
            strata["triggered"].append(pair)

    summary = {name: _summarize(items) for name, items in strata.items()}
    for name, scope in summary.items():
        old = prior_score["strata"][name]
        need(scope["n"] == old["n"] and all(scope["correct"][arm]["targeted"] == old["correct"][arm]["targeted"]
                                            for arm in ARMS), "Prior targeted totals did not reproduce")
        need(scope["routing"]["targeted"] == old["routing"]["targeted"], "Prior targeted pairs did not reproduce")
    return dict(
        status="citation_check_development_scored", release_sha256=release_sha256,
        prior_release_sha256=prior_release_sha256, prior_score_sha256=prior_score_sha256,
        development_labels=N, confirmation_labels=0, clue_labels=0, gpu=0,
        strata=summary, changes=changes,
        by_gate={
            arm: {
                gate: dict(n=len(items), targeted_correct=sum(answer["targeted"] for answer, _ in items),
                           verified_correct=sum(answer["verified"] for answer, _ in items),
                           **paired(items, "verified", "targeted"))
                for gate, items in arm_items.items()
            }
            for arm, arm_items in by_gate.items()
        },
        physical_claims=_claim_summary(unique_claims),
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
