"""Reconstruct the committed development inputs and decisions without loading weights."""

from collections import Counter
from dataclasses import asdict
import json
from math import prod
import os
from pathlib import Path
import sys

from . import policy, receiver
from .experiment import MODES, NATIVE_REPORT_INSTRUCTION, plan_actions, run_factorial, targeted_view
from .live_run import ACTION, PREFLIGHT, ROOT, digest, emit, files, load_history, need, write
from .native_runtime import QA_TEMPLATE, _receipt
from .target import target_from_question


def read(path, pin=None):
    if pin is not None:
        need(digest(path) == pin, f"Changed receipt: {path.name}")
    return json.loads(path.read_bytes())


def sources(old, native, row, work):
    reports = []
    for kind, initial in zip(("source_shared", "source_query"), row.requests, strict=True):
        proof = read(work / kind / "outputs" / f"{kind}.json")
        need(proof["audit_passed"] and proof["actual_model_generation"] and proof["stage"] == "source"
             and proof["visual_calls"] == 1 and proof["visual_input"]
             and all(proof[k] == initial[k] for k in old.SOURCE_FIELDS), "Source input changed")
        reports.append(proof)
    messages, tools = old.request(row.contract, row.case, row.media, *reports)
    plan = read(work / "plan" / "outputs" / "live_text_plan.json")
    need(plan["audit_passed"] and plan["actual_model_generation"] and plan["stage"] == "planner"
         and plan["visual_calls"] == 0 and not plan["visual_input"] and not plan["images"]
         and plan["messages"] == messages and plan["tools"] == tools, "Text Planner input changed")
    parsed = row.contract.plan_record(plan, row.media["duration"], row.media)
    return reports, parsed["arguments"] if parsed["valid"] else None


def validate_route(s, row, processor, rope, reports, text, target, foil, trace, eligible):
    grid, = {tuple(g) for g in reports[0]["image_grid_thw"]}
    cap = row.requests[0].get("max_input_tokens", 20000)
    inputs = [r["text"] for r in reports]
    actual = receiver.render(s, processor, rope, grid, target, text["query"], inputs, cap)[0]
    need(trace["actual_prompt"] == actual, "Live receiver inputs changed")
    scan = trace["scan"]
    need(scan["forwards"] == 33 and scan["visual_forwards"] == 0, "Wrong receiver scan budget")
    candidate = policy.nominate(scan["real"], scan["zero"], eligible)
    if candidate is None:
        need(all(trace[k] is None for k in ("address", "wrong_prompt", "controls", "decision")),
             "Receiver routed without a candidate")
        return trace
    wrong = receiver.render(s, processor, rope, grid, foil, text["query"], inputs, cap)[0]
    control = trace["controls"]
    need(trace["wrong_prompt"] == wrong and control["forwards"] == 3 and control["visual_forwards"] == 0,
         "Wrong-target layout or control budget changed")
    decision = policy.approve(candidate, scan["zero"], control["scrambled"], control["wrong"], control["wrong_zero"])
    need(asdict(decision) == trace["decision"] and trace["address"] == (candidate.address if decision.accepted else None),
         "Receiver approval did not reproduce")
    return trace


def focus_call(work, record, mode, arm, indices, text, target, fps, budget_signatures):
    row = record["focus"][mode]["rows"][arm]
    need(row["indices"] == indices and row["calls"] == 1, "Focus replay changed the physical request")
    directory = work / "followup" / "focus" / mode / arm
    ref = row["record"]
    receipt = read(directory / ref["path"], ref["sha256"])
    messages = receipt["messages"]
    proof = _receipt(directory, ref, "focus", f"focus_{arm}", messages, 2048)
    body = messages[0]["content"]
    query = "Question:\n" + text["query"] + "\n\n"
    need(len(messages) == 1 and messages[0]["role"] == "user"
         and body[1:-1:2] == [dict(type="text", text=z) for z in row["clocks"]]
         and body[2:-1:2] == [dict(type="image")] * len(indices)
         and query in body[-1]["text"] and row["observation"] == proof["text"]
         and row["capped"] == proof["generation_cap"], "Focus changed its input or report")
    if mode == "native":
        need(NATIVE_REPORT_INSTRUCTION in body[-1]["text"], "Original reporting instruction was lost")
    else:
        prefix, marker, _ = body[-1]["text"].rpartition(query)
        need(marker, "Target report lost the original query")
        source = {k: row[k] for k in ("indices", "clocks", "jpeg_sha256", "rgb_sha256")}
        original = [dict(z) for z in body]
        original[-1]["text"] = prefix + query.removesuffix("\n\n") + NATIVE_REPORT_INSTRUCTION + "native"
        source["messages"] = [dict(role="user", content=original)]
        need(targeted_view(source, target, fps)["messages"] == messages, "Targeted instruction changed")
    images = proof["images"]
    grids = proof["image_grid_thw"]
    patches, visual = sum(map(prod, grids)), sum(proof["mm_token_type_ids"])
    need(len(indices) == len(images) == len(grids) and patches == proof["pixel_values_shape"][0]
         and 0 < visual < proof["input_tokens"]
         and [image["sha256"] for image in images] == row["jpeg_sha256"], "Focus image budget or identity changed")
    for image in images:
        need(image["path"] == f"images/{image['sha256']}.jpg"
             and digest(directory / image["path"]) == image["sha256"], "Bound Focus JPEG changed")
    info = record["visual_ledger"][mode][arm]
    expected = dict(
        record=ref, frames=len(indices), patches=patches, visual_tokens=visual, input_tokens=proof["input_tokens"],
        nonvisual_input_tokens=proof["input_tokens"] - visual, output_tokens=proof["output_tokens"],
        image_grid_thw=grids, pixel_values_shape=proof["pixel_values_shape"],
        pixel_values_sha256=proof["pixel_values_sha256"],
    )
    need(info == expected, "Changed token ledger")
    same_view = (row["clocks"], row["jpeg_sha256"], row["rgb_sha256"],
                 grids, proof["pixel_values_shape"], proof["pixel_values_sha256"])
    need(budget_signatures.setdefault(tuple(indices), same_view) == same_view, "Matched visual model input changed")
    return row


def qa_call(native, runtime, case, work, record, mode, arm, messages, tokenizer):
    cell = record["cells"][arm][mode]
    expected = cell["qa"]
    need(cell["qa_reused_from"] is None, "Physical QA cannot be a reused cell")
    directory = work / "followup" / "qa" / mode / arm
    key = f"qa_{arm}"
    proof = _receipt(directory, expected["record"], "qa", key, messages, runtime.COMMON["qa_cap"])
    prompt = tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
    need(not proof["images"] and proof["rendered"] == prompt
         and proof["input_ids"] == tokenizer.encode(prompt, add_special_tokens=False),
         "Changed pure-text QA input")
    outcome = dict(
        status="generated", record=expected["record"], capped=proof["generation_cap"],
        answer=None if proof["generation_cap"] else native.option(proof["text"], case["choices"]),
        input_tokens=proof["input_tokens"], output_tokens=proof["output_tokens"],
    )
    need(outcome == expected, "QA result does not match its native generation")
    return outcome


def audit_case(old, d, row, ordinal, work, record, manifest, processor, rope, deps):
    from .bound_action import action_contract
    from .gap_menu import gaps

    split, triad, runtime, io, native = deps
    foil = manifest["foils"][row.cid]
    need(record["cid"] == row.cid and record["ordinal"] == ordinal and record["case_uid"] == row.case["video_uid"]
         and record["foil_id"] == foil["foil_id"] and record["source_check"] is None, "Wrong committed case")
    reports, text = sources(old, native, row, work)
    target = target_from_question(row.case["question"])
    need(text == record["text_action"] and asdict(target) == record["target"], "Tool query or evidence target changed")
    geo, sampler, _ = old.bridge.metadata(row.cid, row.media, row.job, row.case, native, d.slot)
    trace = record["route_audit"]
    action, sample, actual = plan_actions(
        split, triad, action_contract, gaps, text, geo, sampler,
        lambda eligible: validate_route(d.s, row, processor, rope, reports, text, target,
                                        target_from_question(foil["question"]), trace, eligible),
    )
    need(actual == trace and action == record["routes"], "Native action did not reproduce")
    signatures = {}
    for mode in MODES:
        triad.audit(record["focus"][mode], action, ordinal, sample, text)
        measured = record["visual_ledger"][mode]
        need(set(measured) == {arm for arm, v in record["focus"][mode]["rows"].items() if v["calls"]},
             "Unexpected physical Focus ledger")
    replay = run_factorial(
        triad, io, native, row.case, [x["text"] for x in reports], target, geo["fps"], ordinal, action, sample, text,
        lambda mode, arm, indices, raw: focus_call(work, record, mode, arm, indices, raw,
                                                   target, geo["fps"], signatures),
        lambda mode, arm, messages: qa_call(native, runtime, row.case, work, record, mode, arm,
                                             messages, processor.tokenizer),
    )
    need(all(record[k] == v for k, v in replay.items()), "Paired execution or QA reuse changed")
    budgets = {
        (v["frames"], v["patches"], v["visual_tokens"])
        for condition in record["visual_ledger"].values() for v in condition.values()
    }
    need(len(budgets) == int(text is not None), "Visual budgets differ across the six cells")
    return dict(
        cases=1, focus=sum(x["physical_calls"] for x in record["focus"].values()),
        qa=record["unique_qa_calls"], invalid_action=int(text is None),
        invalid_qa_cells=sum(int(v["qa"] is not None and v["qa"]["answer"] is None)
                             for cell in record["cells"].values() for v in cell.values()),
        capped_focus=sum(int(v.get("capped", False)) for ledger in record["focus"].values()
                         for v in ledger["rows"].values() if v["calls"]),
    )


def audit(*, prefix=False):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU audit requires hidden GPUs")
    manifest = read(PREFLIGHT)
    old, d, order = load_history()
    need(manifest["files"] == files() and order == manifest["order"], "Changed frozen execution inputs")
    path = ACTION / "terminal.json"
    terminal = None if prefix else read(path)
    if terminal is not None:
        pid = str(terminal["pid"])
        need(terminal["status"] == "complete" and terminal["order"] == order
             and terminal["cases"] == len(terminal["rows"]) == len(order)
             and terminal["pin"] == digest(PREFLIGHT) and terminal["labels"] == terminal["holdout"] == 0,
             "Experiment must complete before release")
        proc = Path("/proc") / pid / "stat"
        need((not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z")
             and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(),
             "Experiment process still owns resources")
    import run_itt_utility as native

    deps = (*old.bridge.attach_code(), native)
    s = d.s
    s.torch.set_num_threads(2)
    m = s.old.n.MODEL
    processor = s.prepare_native(s.AutoProcessor.from_pretrained(m, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(m, local_files_only=True))
    totals = Counter()
    for i, row in enumerate(d.rows):
        work = ACTION / f"case_{i:02d}"
        case = work / "case.json"
        if prefix and not case.exists():
            continue
        record = read(case, terminal["rows"][row.cid] if terminal is not None else None)
        totals.update(audit_case(old, d, row, i, work, record, manifest, processor, rope, deps))
    need(not s.torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    if terminal is None:
        emit(status="missing_fact_prefix_audited", **totals)
        return
    need(totals["cases"] == len(order) and totals["focus"] == terminal["counts"]["focus_calls"]
         and totals["qa"] == terminal["counts"]["unique_qa"], "Committed resource totals changed")
    record = dict(status="missing_fact_factorial_audited", terminal=digest(path), gpu_released=True,
                  peak_bytes=terminal["peak_bytes"], labels=0, holdout=0, training=0, **totals)
    pin = write(ROOT / "release.json", record)
    emit(release_sha256=pin, **record)


if __name__ == "__main__":
    args = sys.argv[1:]
    need(args in (["prefix"], ["release"]), "prefix | release")
    audit(prefix=args == ["prefix"])
