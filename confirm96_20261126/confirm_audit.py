"""Independent label-free CPU audit for the 96-video confirmation run."""

from collections import Counter
from dataclasses import asdict
import json
from math import prod
import os
from pathlib import Path
import sys
from types import SimpleNamespace

from entropy_routing_20261125 import multi_action, policy, zoom_verify_run as zr
from entropy_routing_20261125.experiment import plan_actions
from missing_fact_routing_20261124 import receiver
from missing_fact_routing_20261124.live_run import digest, emit, need, write
from missing_fact_routing_20261124.native_runtime import SHA256, _receipt
from missing_fact_routing_20261124.target import target_from_question

from .confirm_inputs import N_CONFIRM, abi_case
from .confirm_run import (
    ACTION, ARMS, MAX_INPUT_TOKENS, PLANNER_CAP, PREFLIGHT, ROOT, SOURCE_CAP,
    _verify_qa_receipt, files, load_abi_modules, make_sampler, source_messages,
)


def read(path, pin=None):
    if pin is not None:
        need(digest(path) == pin, f"Changed receipt: {Path(path).name}")
    return json.loads(Path(path).read_bytes())


def _verify_source_receipt(directory, ref, key, expected_messages, cap):
    """Verify a native visual Source receipt (visual_input=True, visual_calls=1, 32 JPEGs on disk)."""
    relative = f"outputs/{key}.json"
    need(ref.get("path") == relative and isinstance(ref.get("sha256"), str)
         and SHA256.fullmatch(ref["sha256"]) is not None, "Wrong physical Source reference")
    path = directory / relative
    need(not path.is_symlink() and not path.parent.is_symlink(), "Unexpected linked Source receipt")
    proof = read(path, ref["sha256"])
    need(
        proof["key"] == key and proof["stage"] == "source"
        and proof["audit_passed"] is True and proof["audit_pending"] is False
        and proof["actual_model_generation"] is True and proof["visual_input"] is True
        and proof["visual_calls"] == 1 and proof["latent_present"] is False
        and proof["tools"] is None and proof["messages"] == expected_messages
        and proof["maximum_new_tokens"] == cap
        and isinstance(proof["text"], str) and len(proof["input_ids"]) == proof["input_tokens"] > 0
        and 0 < len(proof["generated_token_ids"]) == proof["output_tokens"] <= cap
        and proof["generation_cap"] is (proof["output_tokens"] == cap)
        and len(proof["images"]) == 32,
        f"Source visual receipt mismatch: {key}",
    )
    for img in proof["images"]:
        ipath = directory / img["path"]
        need(not ipath.is_symlink() and digest(ipath) == img["sha256"], f"Source JPEG changed: {key}")
    return proof


def audit_one_case(d, case, media, foil_info, ordinal, work, record, processor, rope, deps):
    native, contract, split, _, _, io = deps
    cid = case["id"]
    case_abi = abi_case(case)
    fps, frame_count, duration = media["fps"], media["frame_count"], media["duration"]
    overview_indices, overview_times = media["overview_indices"], media["overview_times"]

    need(record["cid"] == cid and record["rank"] == case["rank"] and record["ordinal"] == ordinal
         and record["qid"] == case["qid"] and record["case_uid"] == case["video_uid"]
         and record["split"] == "confirmation" and record["foil_id"] == foil_info["foil_id"],
         f"Case identity mismatch: {cid}")

    # 1. Audit both live Source receipts
    src_recs = {}
    for kind, dir_name in (("shared", "source_shared"), ("query_specific", "source_query")):
        sdir = work / dir_name
        ref = record["sources"][dir_name]
        expected_msgs = source_messages(native, case_abi, overview_times, kind)
        src_recs[dir_name] = _verify_source_receipt(sdir, ref, dir_name, expected_msgs, SOURCE_CAP)
    need(src_recs["source_shared"]["pixel_values_sha256"] == src_recs["source_query"]["pixel_values_sha256"]
         and src_recs["source_shared"]["image_grid_thw"] == src_recs["source_query"]["image_grid_thw"],
         f"Source visual mismatch: {cid}")
    reports = [src_recs["source_shared"]["text"], src_recs["source_query"]["text"]]

    # 2. Audit pure-text live Planner
    pmessages_all, ptools = contract.planning_messages(case_abi, reports, duration, overview_times)
    pdir = work / "plan"
    pref = record["planner_receipt"]
    pproof = read(pdir / pref["path"], pref["sha256"])
    need(pproof["key"] == "live_text_plan" and pproof["stage"] == "planner"
         and pproof["audit_passed"] is True and pproof["actual_model_generation"] is True
         and pproof["visual_calls"] == 0 and not pproof["visual_input"] and not pproof["images"]
         and pproof["messages"] == pmessages_all["T_plus"] and pproof["tools"] == ptools
         and pproof["maximum_new_tokens"] == PLANNER_CAP, f"Planner receipt mismatch: {cid}")
    parsed_plan = contract.plan_record(pproof, duration, media)
    expected_text = parsed_plan["arguments"] if parsed_plan["valid"] else None
    need(record["text_action"] == expected_text, f"Parsed text action mismatch: {cid}")

    # 3. Audit calibrated-lift entropy routing + controls
    target = target_from_question(case["question"])
    foil_target = target_from_question(foil_info["question"])
    need(asdict(target) == record["target"], f"Target mismatch: {cid}")
    sample = make_sampler(native, fps, frame_count)
    geometry = dict(source=overview_indices, fps=fps, count=frame_count)
    trace = record["route_audit"]

    def validate_route(eligible):
        grid, = {tuple(sz) for sz in src_recs["source_shared"]["image_grid_thw"]}
        actual = receiver.render(d.s, processor, rope, grid, target, expected_text["query"], reports, MAX_INPUT_TOKENS)[0]
        need(trace["actual_prompt"] == actual, f"Receiver prompt mismatch: {cid}")
        scan = trace["scan"]
        need(scan["forwards"] == 33 and scan["visual_forwards"] == 0, f"Receiver scan budget mismatch: {cid}")
        cand = policy.nominate(scan["real"], scan["zero"], eligible)
        if cand is None:
            need(trace["addresses"] == [] and trace["per_gap_frames"] == 0
                 and all(trace[k] is None for k in ("wrong_prompt", "controls", "decision")),
                 f"Unexpected route without nomination: {cid}")
            return trace
        wrong = receiver.render(d.s, processor, rope, grid, foil_target, expected_text["query"], reports, MAX_INPUT_TOKENS)[0]
        ctrl = trace["controls"]
        need(trace["wrong_prompt"] == wrong and ctrl["forwards"] == 3 and ctrl["visual_forwards"] == 0,
             f"Control prompt/budget mismatch: {cid}")
        dec = policy.approve(cand, scan["real"], scan["zero"], ctrl["scrambled"], ctrl["wrong"], ctrl["wrong_zero"], eligible)
        exp_dec = dict(asdict(dec), addresses=list(dec.addresses))
        need(exp_dec == trace["decision"]
             and trace["addresses"] == (list(dec.addresses) if dec.accepted else [])
             and trace["per_gap_frames"] == (dec.per_gap_frames if dec.accepted else 0),
             f"Entropy approval mismatch: {cid}")
        return trace

    action, bounded_sampler, actual_trace = plan_actions(split, expected_text, geometry, sample, validate_route)
    need(actual_trace == trace and action == record["routes"], f"Multi-clip routes mismatch: {cid}")

    # 4. Audit Focus-1 (native) & exact 32-frame visual budget parity
    multi_action.audit(record["focus1"], action, ordinal, bounded_sampler, expected_text)
    f1_budgets = set()
    f1_capped = 0
    for arm, frow in record["focus1"]["rows"].items():
        if frow.get("calls", 0) != 1:
            continue
        fdir = work / "focus1" / arm
        ref = frow["record"]
        raw = read(fdir / ref["path"], ref["sha256"])
        proof = _receipt(fdir, ref, "focus", f"focus_{arm}", raw["messages"], SOURCE_CAP)
        need(("Question:\n" + expected_text["query"] + "\n\n") in proof["messages"][0]["content"][-1]["text"]
             and proof["text"] == frow["observation"] and proof["generation_cap"] == frow["capped"],
             f"Focus-1 prompt/output mismatch: {cid}/{arm}")
        grids, pixels = proof["image_grid_thw"], proof["pixel_values_shape"]
        patches, visual = sum(map(prod, grids)), sum(proof["mm_token_type_ids"])
        need(len(grids) == len(proof["images"]) == len(frow["indices"]) and patches == pixels[0]
             and [img["sha256"] for img in proof["images"]] == frow["jpeg_sha256"],
             f"Focus-1 JPEG receipt mismatch: {cid}/{arm}")
        for img in proof["images"]:
            need(digest(fdir / img["path"]) == img["sha256"], f"Focus-1 JPEG modified: {cid}/{arm}")
        f1_budgets.add((len(frow["indices"]), patches, visual))
        f1_capped += int(frow["capped"])
    need(len(f1_budgets) == int(expected_text is not None), f"Focus-1 visual budgets differed: {cid}")

    # 5. Audit QA-1
    arms_order = multi_action.order_for(ordinal)
    qa_cache = {}
    phys_qa1 = 0
    inv_qa = 0
    for arm in arms_order:
        cell = record["qa1"][arm]
        obs1 = record["focus1"]["rows"][arm]["observation"]
        if obs1 is None:
            need(cell["status"] == "common_invalid" and cell["qa"] is None, f"Bad common invalid QA-1: {cid}/{arm}")
            continue
        need(cell["status"] == "native" and cell["observation"] == obs1, f"QA-1 obs mismatch: {cid}/{arm}")
        msgs = io.qa_messages(native, case_abi, reports, obs1)
        mkey = json.dumps(msgs, ensure_ascii=False, sort_keys=True, allow_nan=False)
        if mkey in qa_cache:
            exp_origin, exp_qa = qa_cache[mkey]
            need(cell["qa_reused_from"] == exp_origin and cell["qa"] == exp_qa, f"Bad QA-1 reuse: {cid}/{arm}")
        else:
            need(cell["qa_reused_from"] is None, f"Unexpected QA-1 reuse: {cid}/{arm}")
            verified = _verify_qa_receipt(work / "qa1" / arm, arm, msgs, case["choices"],
                                          native, processor.tokenizer, cell["qa"]["record"])
            need(verified == cell["qa"], f"QA-1 receipt mismatch: {cid}/{arm}")
            qa_cache[mkey] = (dict(stage="qa1", arm=arm), verified)
            phys_qa1 += 1
        inv_qa += int(cell["qa"]["answer"] is None)
    need(phys_qa1 == record["unique_qa1_calls"], f"QA-1 count mismatch: {cid}")

    # 6. Audit Focus-2 plan, receipts, and QA-2
    expected_f2_plan = zr.plan_case_zoom(
        SimpleNamespace(media=media),
        dict(text_action=expected_text, routes=action, cells={a: dict(native=record["qa1"][a]) for a in ARMS}),
    )
    need(record["focus2_plan"] == expected_f2_plan, f"Focus-2 plan mismatch: {cid}")
    phys_f2 = 0
    phys_qa2 = 0
    f2_capped = 0
    if expected_f2_plan["triggered"]:
        f2_budgets = set()
        for arm in arms_order:
            frow = record["focus2"][arm]
            if frow["calls"] == 0:
                src = frow["reused_from"]
                need(expected_f2_plan["arms"][arm]["indices"] == expected_f2_plan["arms"][src]["indices"]
                     and frow["observation"] == record["focus2"][src]["observation"], f"Bad Focus-2 reuse: {cid}/{arm}")
                continue
            need(frow["calls"] == 1 and frow["indices"] == expected_f2_plan["arms"][arm]["indices"],
                 f"Focus-2 indices mismatch: {cid}/{arm}")
            f2dir = work / "focus2" / arm
            ref = frow["record"]
            raw = read(f2dir / ref["path"], ref["sha256"])
            proof = _receipt(f2dir, ref, "focus", f"focus_{arm}", raw["messages"], SOURCE_CAP)
            need(("Question:\n" + expected_text["query"] + "\n\n") in proof["messages"][0]["content"][-1]["text"]
                 and proof["text"] == frow["observation"] and proof["generation_cap"] == frow["capped"],
                 f"Focus-2 prompt/output mismatch: {cid}/{arm}")
            grids, pixels = proof["image_grid_thw"], proof["pixel_values_shape"]
            patches, visual = sum(map(prod, grids)), sum(proof["mm_token_type_ids"])
            need(len(grids) == len(proof["images"]) == zr.ZOOM_FRAMES and patches == pixels[0]
                 and [img["sha256"] for img in proof["images"]] == frow["jpeg_sha256"],
                 f"Focus-2 JPEG receipt mismatch: {cid}/{arm}")
            for img in proof["images"]:
                need(digest(f2dir / img["path"]) == img["sha256"], f"Focus-2 JPEG modified: {cid}/{arm}")
            f2_budgets.add((zr.ZOOM_FRAMES, patches, visual))
            phys_f2 += 1
            f2_capped += int(frow["capped"])
        need(len(f2_budgets) == 1 and phys_f2 == record["physical_focus2_calls"], f"Focus-2 budget mismatch: {cid}")

        for arm in arms_order:
            cell = record["qa2"][arm]
            base_obs = record["qa1"][arm]["observation"]
            base_ans = (record["qa1"][arm]["qa"] or {}).get("answer")
            frow = record["focus2"][arm]
            exp_obs = zr.combine_observation(base_obs, frow["observation"], frow["capped"])
            need(cell["observation"] == exp_obs and cell["kind"] == expected_f2_plan["arms"][arm]["kind"],
                 f"QA-2 obs/kind mismatch: {cid}/{arm}")
            msgs = io.qa_messages(native, case_abi, reports, exp_obs)
            mkey = json.dumps(msgs, ensure_ascii=False, sort_keys=True, allow_nan=False)
            if mkey in qa_cache:
                exp_origin, exp_qa = qa_cache[mkey]
                need(cell["qa_reused_from"] == exp_origin and cell["qa"] == exp_qa, f"Bad QA-2 reuse: {cid}/{arm}")
            else:
                need(cell["qa_reused_from"] is None, f"Unexpected QA-2 reuse: {cid}/{arm}")
                verified = _verify_qa_receipt(work / "qa2" / arm, arm, msgs, case["choices"],
                                              native, processor.tokenizer, cell["qa"]["record"])
                need(verified == cell["qa"], f"QA-2 receipt mismatch: {cid}/{arm}")
                qa_cache[mkey] = (dict(stage="qa2", arm=arm), verified)
                phys_qa2 += 1
            exp_eq = cell["qa"]["answer"]
            exp_op = exp_eq if cell["kind"] == "on_policy" else base_ans
            need(cell["equal_budget_answer"] == exp_eq and cell["on_policy_answer"] == exp_op,
                 f"QA-2 endpoint mismatch: {cid}/{arm}")
            inv_qa += int(cell["qa"]["answer"] is None)
    else:
        need(record["physical_focus2_calls"] == 0 and record["unique_qa2_calls"] == 0, f"Untriggered Focus-2 spent calls: {cid}")
        for arm in arms_order:
            base_ans = (record["qa1"][arm]["qa"] or {}).get("answer")
            cell = record["qa2"][arm]
            need(cell["equal_budget_answer"] == base_ans and cell["on_policy_answer"] == base_ans
                 and cell["qa"] is None and cell["f2_observation"] is None,
                 f"Untriggered QA-2 mismatch: {cid}/{arm}")
    need(phys_qa2 == record["unique_qa2_calls"], f"QA-2 count mismatch: {cid}")

    return dict(
        cases=1,
        invalid_action=int(expected_text is None),
        focus1_calls=record["focus1"]["physical_calls"],
        qa1_calls=phys_qa1,
        f2_triggered=int(expected_f2_plan["triggered"]),
        focus2_calls=phys_f2,
        qa2_calls=phys_qa2,
        capped_focus1=f1_capped,
        capped_focus2=f2_capped,
        invalid_qa_cells=inv_qa,
    )


def audit(*, prefix=False):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU audit requires hidden GPUs")
    manifest = read(PREFLIGHT)
    need(manifest["status"] == "confirm96_preflight_frozen" and manifest["files"] == files()
         and manifest["labels"] == manifest["clues"] == manifest["training"] == manifest["gpu"] == 0
         and len(manifest["cases"]) == N_CONFIRM, "Changed confirmation manifest")
    old, d, _, native, contract, split, triad, runtime, io, _, _ = load_abi_modules()

    tpath = ACTION / "terminal.json"
    terminal = None if prefix else read(tpath)
    if terminal is not None:
        pid = str(terminal["pid"])
        need(terminal["status"] == "complete" and terminal["split"] == "confirmation"
             and terminal["order"] == manifest["order"] and terminal["cases"] == N_CONFIRM == len(terminal["rows"])
             and terminal["pin"] == digest(PREFLIGHT)
             and terminal["labels"] == terminal["clues"] == terminal["training"] == 0,
             "Confirmation run must complete before release")
        proc = Path("/proc") / pid / "stat"
        need((not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z")
             and pid not in old.smi(terminal["gpu"], "--query-compute-apps=pid").split(),
             "GPU process still running")

    s = d.s
    s.torch.set_num_threads(2)
    m = s.old.n.MODEL
    processor = s.prepare_native(s.AutoProcessor.from_pretrained(m, local_files_only=True), d.nat)
    rope = s.rope(s.AutoConfig.from_pretrained(m, local_files_only=True))
    deps = (native, contract, split, triad, runtime, io)

    totals = Counter()
    for ordinal, case in enumerate(manifest["cases"]):
        cid = case["id"]
        work = ACTION / f"case_{ordinal:02d}"
        cpath = work / "case.json"
        if prefix and not cpath.exists():
            continue
        rec = read(cpath, terminal["rows"][cid] if terminal is not None else None)
        totals.update(audit_one_case(d, case, manifest["media"][cid], manifest["foils"][cid],
                                     ordinal, work, rec, processor, rope, deps))
    need(not s.torch.cuda.is_initialized(), "CPU audit initialized CUDA")
    if prefix:
        emit(status="confirm96_prefix_audited", **totals)
        return
    need(totals["cases"] == N_CONFIRM
         and totals["focus1_calls"] == terminal["counts"]["focus1_calls"]
         and totals["qa1_calls"] == terminal["counts"]["qa1_calls"]
         and totals["focus2_calls"] == terminal["counts"]["focus2_calls"]
         and totals["qa2_calls"] == terminal["counts"]["qa2_calls"],
         "Audited totals differ from terminal")
    record = dict(
        status="confirm96_development_free_audited",
        split="confirmation",
        terminal=digest(tpath),
        gpu_released=True,
        peak_bytes=terminal["peak_bytes"],
        labels=0,
        clues=0,
        training=0,
        **totals,
    )
    pin = write(ROOT / "release.json", record)
    emit(release_sha256=pin, **record)


if __name__ == "__main__":
    args = sys.argv[1:]
    need(args in (["prefix"], ["release"]), "prefix | release")
    audit(prefix=args == ["prefix"])
