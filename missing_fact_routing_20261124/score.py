"""Score the audited LAVE development factorial without opening confirmation labels."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations
import json
from math import comb
import os
from pathlib import Path


N = 51
ARMS = ("text", "state_split", "clock_split")
MODES = ("native", "targeted")
CLOSED = dict(labels=0, holdout=0, training=0)
ANNOTATION_PIN = "f2db9d2e8144e30806c3eab3ea50e7e1681bf9dd31d0dd6ce2871f960559de12"
QID_PIN = "d5161c75a0f505032fc53b95d41b5a54bcb52d319cc9f11020f89e5aca5d90fe"
SCANNER_PIN = "7b772076a62e55e3da090dc3bc4d6e62223b3a8716bb78f75b2d98684ac9cf20"
IDENTITY_PIN = "8cac07a76593f89a9d43b58d660711374f8426369fb56666d8fb0918f6761d8c"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def pinned(path, pin):
    raw = Path(path).read_bytes()
    need(sha256(raw).hexdigest() == pin, f"Changed input bytes: {Path(path).name}")
    return raw


def closed(receipt):
    return all(type(receipt.get(key)) is int and receipt[key] == expected for key, expected in CLOSED.items())


def physical_answer(work, arm, mode, cell, choices, native_option):
    item = cell["qa"]
    ref = item["record"]
    key = f"qa_{arm}"
    rel = f"outputs/{key}.json"
    need(ref["path"] == rel, "QA reference is outside its physical cell")
    path = work / "followup" / "qa" / mode / arm / rel
    raw = json.loads(pinned(path, ref["sha256"]))
    cap = raw["generation_cap"]
    need(raw["stage"] == "qa" and raw["key"] == key and raw["audit_passed"] is True
         and raw["audit_pending"] is False and raw["actual_model_generation"] is True
         and raw["visual_input"] is False and raw["visual_calls"] == 0 and not raw["images"]
         and raw["latent_present"] is False and type(cap) is bool and isinstance(raw["text"], str),
         "QA receipt does not describe an audited native text generation")
    answer = None if cap else native_option(raw["text"], choices)
    expected = dict(answer=answer, capped=cap, record=ref, status="generated",
                    input_tokens=raw["input_tokens"], output_tokens=raw["output_tokens"])
    need(item == expected, "Normalized QA answer or token counts disagree with the generation")
    return answer


def case_outcomes(work, row, ordinal, record, native_option):
    need(record["cid"] == row.cid and record["case_uid"] == row.case["video_uid"]
         and record["ordinal"] == ordinal, "Case does not match the frozen development row")
    legal = record["text_action"] is not None
    routes = record["routes"]["indices"]
    arms = record["focus"]["native"]["execution_order"]
    need(len(arms) == len(ARMS) and set(arms) == set(routes) == set(record["cells"]) == set(ARMS),
         "Incomplete paired routing arms")
    route_ids = {arm: tuple(routes[arm]) for arm in ARMS}
    need(all(bool(indices) == legal for indices in route_ids.values()), "Common invalid action changed")
    modes = MODES if (ordinal // 6) % 2 == 0 else MODES[::-1]
    seen, results = {}, {}
    invalid_cell = dict(status="common_invalid", observation=None, qa=None)
    for arm in arms:
        need(set(record["cells"][arm]) == set(MODES), "Incomplete paired reporting modes")
        for mode in modes:
            cell = record["cells"][arm][mode]
            if not legal:
                need(cell == invalid_cell, "Common invalid action produced an answer")
                results[arm, mode] = (None, cell["status"], "common_invalid", False)
                continue
            observation = cell["observation"]
            status = cell["status"]
            report_states = {"native"} if mode == "native" else {"observed", "unresolved", "invalid"}
            need(isinstance(observation, str) and status in report_states and type(cell["capped"]) is bool,
                 "Wrong serialized report state")
            reuse = cell["qa_reused_from"]
            # All other QA fields are fixed within this audited case. Equal
            # observations therefore have byte-identical native QA prompts.
            if observation in seen:
                owner, first_qa, answer = seen[observation]
                need(reuse == owner and cell["qa"] == first_qa, "Identical QA does not reuse its first generation")
            else:
                need(reuse is None, "QA was reused with a different observation")
                answer = physical_answer(work, arm, mode, cell, row.case["choices"], native_option)
                seen[observation] = (dict(arm=arm, mode=mode), cell["qa"], answer)
            outcome = "capped" if cell["qa"]["capped"] else "invalid_option" if answer is None else "valid"
            results[arm, mode] = (answer, status, outcome, cell["capped"])
    need(record["unique_qa_calls"] == len(seen), "Physical QA count changed")
    return results, route_ids, legal


def project_labels(rows, native_option, annotation_path, qid_selection_path, projection_path):
    selection = json.loads(pinned(qid_selection_path, QID_PIN))
    need(selection["annotation_sha256"] == ANNOTATION_PIN and selection["identity_receipt_sha256"] == IDENTITY_PIN,
         "QID selection is not bound to the exposed cohort")
    qids = {item["video_uid"]: str(item["qid"]) for item in selection["chosen_metadata"]}
    need(len(qids) == len(selection["chosen_metadata"]) == 96, "Exposed development selection changed")
    public = {row.case["video_uid"]: row for row in rows}
    need(set(public) <= qids.keys(), "A development video has no frozen question")
    namespace = {}
    exec(compile(pinned(projection_path, SCANNER_PIN), "frozen_metadata_projection", "exec"), namespace)
    scanner = namespace["Projection"](pinned(annotation_path, ANNOTATION_PIN).decode("utf-8"))
    labels = {}
    for spans in scanner.records():
        uid = scanner.value(spans, "video_uid", {"video_uid"})
        if uid not in public:
            continue
        qid = scanner.value(spans, "qid", {"qid"})
        if type(qid) not in (int, str) or str(qid) != qids[uid]:
            continue
        row = public[uid]
        question = scanner.value(spans, "question", {"question"})
        choices = scanner.value(spans, "choices", {"choices"})
        need(row.cid not in labels and question == row.case["question"] and choices == row.case["choices"],
             "Official development question or option order differs")
        gold = scanner.value(spans, "right_answer", {"right_answer"})
        need(type(gold) is str and native_option(gold, choices) == gold, "Unexpected official option label")
        labels[row.cid] = gold
    need(set(labels) == {row.cid for row in rows} and all(scanner.decoded[field] == N for field in
         ("question", "choices", "right_answer")) and set(scanner.decoded) ==
         {"video_uid", "qid", "question", "choices", "right_answer"}, "Unexpected label projection scope")
    return labels


def paired(results, treatment, reference):
    win = sum(answer[treatment] and not answer[reference] for answer, _ in results)
    loss = sum(answer[reference] and not answer[treatment] for answer, _ in results)
    discordant = win + loss
    p = min(1.0, 2 * sum(comb(discordant, k) for k in range(min(win, loss) + 1)) / 2**discordant)
    n = len(results)
    return dict(win=win, loss=loss, tie=n - discordant, net=win - loss,
                delta=(win - loss) / n if n else None, p_two_sided_exact=p if n else None)


def summarize(results):
    def score(treatment, reference):
        return paired(results, treatment, reference)

    def change(answer, mode, treatment, reference):
        return answer[treatment, mode] - answer[reference, mode]

    routes = (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))
    return dict(
        n=len(results),
        correct={arm: {mode: sum(answer[arm, mode] for answer, _ in results) for mode in MODES} for arm in ARMS},
        routing={
            mode: {
                f"{a}_vs_{b}": dict(score((a, mode), (b, mode)),
                                     distinct_frame_cases=sum(ids[a] != ids[b] for _, ids in results))
                for a, b in routes
            }
            for mode in MODES
        },
        reporting={arm: score((arm, "targeted"), (arm, "native")) for arm in ARMS},
        interaction={
            f"{a}_vs_{b}": dict(sorted(Counter(
                change(answer, "targeted", a, b) - change(answer, "native", a, b) for answer, _ in results
            ).items()))
            for a, b in routes
        },
    )


def score_development(root, release_sha256, *, rows, native_option, annotation_path, qid_selection_path,
                      projection_path):
    """Score all cases as assigned after replay, reading official answers for these rows only."""
    root, rows = Path(root), list(rows)
    release = json.loads(pinned(root / "release.json", release_sha256))
    need(release["status"] == "missing_fact_factorial_audited" and release["gpu_released"] is True
         and release["cases"] == N and closed(release), "Missing completed label-free GPU-release audit")
    terminal = json.loads(pinned(root / "action" / "terminal.json", release["terminal"]))
    cids = [row.cid for row in rows]
    uids = {row.case["video_uid"] for row in rows}
    need(terminal["status"] == "complete" and terminal["cards"] == 1 and closed(terminal)
         and terminal["cases"] == len(cids) == len(set(cids)) == len(uids) == N
         and terminal["order"] == cids and set(terminal["rows"]) == set(cids)
         and all(row.case["split"] == "development" for row in rows), "Wrong audited development roster")

    answers = {}
    flags = {arm: {mode: Counter() for mode in MODES} for arm in ARMS}
    counted = Counter()
    for ordinal, row in enumerate(rows):
        work = root / "action" / f"case_{ordinal:02d}"
        record = json.loads(pinned(work / "case.json", terminal["rows"][row.cid]))
        outcome, ids, legal = case_outcomes(work, row, ordinal, record, native_option)
        answers[row.cid] = (outcome, ids, legal)
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
    strata = {"all": [], "legal": [], "triggered": []}
    for cid, (outcome, routes, legal) in answers.items():
        comparison = ({cell: result[0] == gold[cid] for cell, result in outcome.items()}, routes)
        strata["all"].append(comparison)
        if legal:
            strata["legal"].append(comparison)
        if len(set(routes.values())) > 1:
            strata["triggered"].append(comparison)
    return dict(status="missing_fact_development_scored", release_sha256=release_sha256,
                development_labels=N, confirmation_labels=0, clue_labels=0, gpu=0,
                qa_calls=counted["qa"], invalid_actions=counted["invalid_action"], cells=flags,
                strata={name: summarize(results) for name, results in strata.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release_sha256")
    for name in ("annotation", "qid_selection", "projection"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Development scoring requires hidden GPUs")
    from .live_run import ROOT, load_history

    _, history, _ = load_history()
    import run_itt_utility as native

    output = score_development(ROOT, args.release_sha256, rows=history.rows, native_option=native.option,
                               annotation_path=args.annotation, qid_selection_path=args.qid_selection,
                               projection_path=args.projection)
    need(not history.s.torch.cuda.is_initialized(), "Scoring initialized CUDA")
    print(json.dumps(output, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
