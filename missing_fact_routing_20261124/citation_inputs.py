"""Read cited facts and their exact cached frames from the released LAVE run."""

from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path

from .audit_run import focus_call
from .experiment import MODES
from .native_runtime import _receipt
from .report import FENCE, serialize_focus_report
from .target import target_from_question


ARMS = ("text", "state_split", "clock_split")
SOURCE_FIELDS = ("input_ids_sha256", "position_ids_sha256", "mm_token_type_ids_sha256", "pixel_values_sha256",
                 "images", "messages", "rendered", "maximum_new_tokens", "image_grid_thw", "tools")


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _local_file(root, relative):
    relative = Path(relative)
    _need(not relative.is_absolute() and ".." not in relative.parts, "Receipt path escapes the released run")
    path = root
    for part in relative.parts:
        path = path / part
        _need(not path.is_symlink(), f"Unexpected linked receipt or cache: {path.name}")
    _need(path.is_file(), f"Missing released file: {path.name}")
    return path


def _record(root, relative, pin):
    data = _local_file(root, relative).read_bytes()
    _need(sha256(data).hexdigest() == pin, f"Changed released file: {Path(relative).name}")
    return json.loads(data)


def _source_reports(work, row):
    reports = []
    for name, original in zip(("source_shared", "source_query"), row.requests, strict=True):
        file = _local_file(work, Path(name) / "outputs" / f"{name}.json")
        source = json.loads(file.read_bytes())
        _need(source["key"] == name and source["stage"] == "source" and source["audit_passed"] is True
              and source["audit_pending"] is False and source["actual_model_generation"] is True
              and source["visual_input"] is True and source["visual_calls"] == 1 and source["latent_present"] is False
              and all(source[field] == original[field] for field in SOURCE_FIELDS) and isinstance(source["text"], str),
              "Changed native Source receipt")
        raw = source["raw"]
        _need(raw["path"] == f"raw/{name}.json", "Wrong native Source raw receipt")
        initial = _record(work, Path(name) / raw["path"], raw["sha256"])
        _need(initial["audit_pending"] is True and initial["text"] == source["text"]
              and initial["key"] == name and initial["generated_token_ids"] == source["generated_token_ids"],
              "Source text differs from the original generation receipt")
        reports.append(source["text"])
    return reports


def _qa_context(work, record, case, reports, cell, arm, native, io):
    owner = cell["qa_reused_from"] or dict(mode="targeted", arm=arm)
    mode, arm = owner["mode"], owner["arm"]
    _need(mode in MODES and arm in ARMS, "Wrong QA reuse origin")
    first = record["cells"][arm][mode]
    _need(first["qa_reused_from"] is None and first["qa"] == cell["qa"]
          and first["observation"] == cell["observation"], "QA reuse changed its observation")
    directory = Path("followup") / "qa" / mode / arm
    key = f"qa_{arm}"
    _local_file(work, directory / "outputs" / f"{key}.json")
    messages = io.qa_messages(native, case, reports, cell["observation"])
    proof = _receipt(work / directory, first["qa"]["record"], "qa", key, messages, 16)
    _need(not proof["images"], "Source context is not bound to a text-only QA receipt")


def _focus_view(work, record, arm, row, target, fps, signatures):
    from PIL import Image

    directory = Path("followup") / "focus" / "targeted" / arm
    relative = f"outputs/focus_{arm}.json"
    ref = row["record"]
    _need(ref["path"] == relative, "Targeted receipt does not name its physical origin")
    proof = _record(work, directory / relative, ref["sha256"])
    for image in proof["images"]:
        expected = f"images/{image['sha256']}.jpg"
        _need(image["path"] == expected, "Unexpected cached Focus frame path")
        path = _local_file(work, directory / expected)
        with Image.open(path) as jpeg:
            _need(jpeg.format == "JPEG" and list(jpeg.size) == image["size"], "Cached Focus frame changed")
    focus_call(work, record, "targeted", arm, row["indices"], record["text_action"], target, fps, signatures)
    return work / directory, proof


def _case_inputs(work, record, row, ordinal, native, io):
    native.final_messages(row.case, [])
    target = target_from_question(row.case["question"])
    _need(record["cid"] == row.cid and record["ordinal"] == ordinal and record["case_uid"] == row.case["video_uid"]
          and record["target"] == asdict(target), "Changed committed development row")
    fps, frame_count = row.media["fps"], row.media["frame_count"]
    _need(type(fps) in (int, float) and isfinite(fps) and fps > 0, "Invalid sealed video frame rate")
    reports = _source_reports(work, row)
    focus = record["focus"]["targeted"]
    order = focus["execution_order"]
    measured = record["visual_ledger"]["targeted"]
    _need(len(order) == len(ARMS) and set(order) == set(focus["rows"]) == set(record["cells"]) == set(ARMS)
          and order == record["focus"]["native"]["execution_order"], "Changed reporting arm roster")
    candidates, seen, signatures = {}, {}, {}
    counts = Counter(targeted_cited_cells=0, invalid_action=int(record["text_action"] is None))
    counts.update(focus_calls=sum(ledger["physical_calls"] for ledger in record["focus"].values()),
                  unique_qa=record["unique_qa_calls"])

    for arm in order:
        result, cell = focus["rows"][arm], record["cells"][arm]["targeted"]
        if record["text_action"] is None:
            _need(result == dict(status="common_invalid_action", calls=0, observation=None)
                  and cell == dict(status="common_invalid", observation=None, qa=None),
                  "A common invalid action generated a report")
            continue
        indices = result["indices"]
        _need(indices == record["routes"]["indices"][arm] and 1 <= len(indices) <= 32
              and all(type(i) is int and 0 <= i < frame_count for i in indices) and indices == sorted(set(indices)),
              "Targeted Focus view differs from the committed frame list")
        times = [i / fps for i in indices]
        status, observation = serialize_focus_report(result["observation"], times, capped=result["capped"])
        _need(cell["status"] == status and cell["observation"] == observation and cell["capped"] is result["capped"],
              "Targeted report no longer matches its parser")
        view = tuple(indices)
        if view in seen:
            origin = seen[view]
            _need(result == dict(focus["rows"][origin], calls=0, reused_identical_input=origin),
                  "Logical alias does not refer to the first physical Focus call")
            if status == "observed":
                candidates[origin]["arms"].append(arm)
                counts["targeted_cited_cells"] += 1
            continue

        _need(result["calls"] == 1 and arm in measured, "Missing physical targeted Focus call")
        seen[view] = arm
        directory, proof = _focus_view(work, record, arm, result, target, fps, signatures)
        _qa_context(work, record, row.case, reports, cell, arm, native, io)
        if status != "observed":
            continue
        text = proof["text"].strip()
        fenced = FENCE.fullmatch(text)
        parsed = json.loads(fenced[1] if fenced else text)
        positions = sorted(parsed["frames"])
        images = proof["images"]
        candidates[arm] = dict(
            origin=arm, arms=[arm], target=record["target"], claim=parsed["observation"].strip(),
            observation=observation, citations=positions, frame_indices=indices, images=images,
            cited_indices=[indices[i] for i in positions], cited_images=[images[i] for i in positions],
            image_grid_thw=proof["image_grid_thw"], fps=fps, directory=directory, focus=result["record"],
        )
        counts["targeted_cited_cells"] += 1
    _need(focus["physical_calls"] == len(measured) == len(seen) and set(measured) == set(seen.values()),
          "Changed targeted physical Focus count")
    return dict(cid=row.cid, ordinal=ordinal, qa_case=row.case, reports=reports, candidates=list(candidates.values())), counts


def load_citation_inputs(root, release_sha256, *, rows, native, io, expected_cases=51):
    """Return one entry per exposed case and one candidate per observed physical view."""
    root = Path(root).resolve(strict=True)
    rows = list(rows)
    cids = [row.cid for row in rows]
    uids = {row.case["video_uid"] for row in rows}
    release = _record(root, "release.json", release_sha256)
    _need(release["status"] == "missing_fact_factorial_audited" and release["gpu_released"] is True
          and release["cases"] == expected_cases, "Prior run has no complete release audit")
    terminal = _record(root, "action/terminal.json", release["terminal"])
    _need(terminal["status"] == "complete" and terminal["cards"] == 1 and terminal["order"] == cids
          and len(cids) == len(set(cids)) == len(uids) == terminal["cases"] == expected_cases
          and set(terminal["rows"]) == set(cids) and terminal["peak_bytes"] == release["peak_bytes"] > 0
          and all(r.case["split"] == "development" and r.case["id"] == r.cid for r in rows),
          "Prior terminal does not cover the exposed development cohort")
    for seal in (release, terminal):
        _need(all(type(seal[key]) is int and seal[key] == 0 for key in ("labels", "holdout", "training")),
              "Prior receipt is outside the label-free development scope")
    preflight = _record(root, "preflight.json", terminal["pin"])
    _need(preflight["status"] == "missing_fact_preflight" and preflight["order"] == cids
          and preflight["labels"] == preflight["holdout"] == preflight["gpu"] == 0,
          "Changed prior CPU seal")

    cases, totals = [], Counter()
    for i, row in enumerate(rows):
        relative = Path("action") / f"case_{i:02d}"
        record = _record(root, relative / "case.json", terminal["rows"][row.cid])
        case, counts = _case_inputs(root / relative, record, row, i, native, io)
        cases.append(case)
        totals.update(counts)
    _need(all(totals[key] == terminal["counts"].get(key, 0) for key in totals)
          and totals["focus_calls"] == release["focus"] and totals["unique_qa"] == release["qa"]
          and totals["invalid_action"] == release["invalid_action"], "Released execution counts changed")
    return cases
