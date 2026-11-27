"""CPU single-unseal scorer for Qwen2.5-VL-7B-Instruct single-GPU main experiments (max_steps=8)."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from .holmes_inputs import HOLMES_JSON, sha256_bytes, write_json_canonical
from .mme_inputs import MME_PARQUET
from .run_qwen25vl_main import BENCHMARKS
from .score_holmes import _gpu_released, exact_mcnemar
from .score_lvb import LVB_PARQUET

ROOT = Path(".")
ARMS_S8 = (
    "vanilla_text",
    "vanilla_step4",
    "vanilla_step6",
    "vanilla_latent_qa",
    "vanilla_latent_visonly",
    "lave_route_text",
    "lave_step4",
    "lave_step6",
    "lave_full",
    "lave_full_visonly",
)
PAIRS_S8 = (
    ("lave_full", "vanilla_text"),
    ("lave_full_visonly", "vanilla_text"),
    ("lave_full", "vanilla_latent_qa"),
    ("lave_full_visonly", "vanilla_latent_visonly"),
    ("lave_route_text", "vanilla_text"),
    ("lave_step4", "vanilla_step4"),
    ("lave_step6", "vanilla_step6"),
    ("vanilla_latent_qa", "vanilla_text"),
    ("vanilla_latent_visonly", "vanilla_text"),
    ("lave_full", "lave_route_text"),
)


def summarize_s8(items):
    n = len(items)
    acc = {}
    for arm in ARMS_S8:
        c = sum(1 for row in items if row["ok"].get(arm, False))
        acc[arm] = dict(correct=c, n=n, pct=round(100.0 * c / n, 2) if n else 0.0)
    paired = {}
    for a, b in PAIRS_S8:
        w = sum(1 for row in items if row["ok"].get(a, False) and not row["ok"].get(b, False))
        l = sum(1 for row in items if row["ok"].get(b, False) and not row["ok"].get(a, False))
        paired[f"{a}_vs_{b}"] = dict(
            win=w,
            loss=l,
            net=w - l,
            delta_pp=round(100.0 * (w - l) / n, 2) if n else 0.0,
            p_two_sided_exact=round(exact_mcnemar(w, l), 6),
        )
    return dict(n=n, accuracy=acc, paired=paired)


def load_gold(adapter):
    if adapter == "holmes":
        raw = json.loads(HOLMES_JSON.read_text())
        return {f"{x['video ID']}__{x['Question ID']}": str(x["Answer"]).strip().upper() for x in raw}
    if adapter == "mme":
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(MME_PARQUET), columns=["question_id", "answer"]).to_pydict()
        return {str(q): str(a).strip().upper() for q, a in zip(tbl["question_id"], tbl["answer"])}
    if adapter == "lvb":
        import pyarrow.parquet as pq
        tbl = pq.read_table(str(LVB_PARQUET), columns=["id", "correct_choice"]).to_pydict()
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return {str(cid): letters[int(ch)] for cid, ch in zip(tbl["id"], tbl["correct_choice"])}
    raise ValueError(adapter)


def score_suite(suite, max_steps=8, check_gpu_release=True):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Score on CPU with CUDA_VISIBLE_DEVICES=''")
    spec = BENCHMARKS[suite]
    pfile = ROOT / spec["manifest"]
    pin = sha256_bytes(pfile.read_bytes())
    if spec["expected_pin"] is not None and pin != spec["expected_pin"]:
        raise RuntimeError(f"Pin mismatch for {suite}")
    manifest = json.loads(pfile.read_bytes())
    cases = manifest["cases"][: spec["limit"]]

    case_dir = ROOT / "action" / f"{suite}_s{max_steps}"
    tpath = case_dir / "terminal.json"
    term = json.loads(tpath.read_bytes()) if tpath.exists() else None
    if term is not None:
        if term["status"] != "complete" or term["pin"] != pin or term["labels"] != 0:
            raise RuntimeError("Terminal receipt invalid")
        if check_gpu_release and not _gpu_released(term["pid"]):
            raise RuntimeError(f"GPU process {term['pid']} still alive")

    gold_map = load_gold(spec["adapter"])
    items = []
    rerouted_items = []
    by_bucket = defaultdict(list)
    by_cat = defaultdict(list)
    stats = Counter()
    invalid_ans = Counter()

    for ordinal, case in enumerate(cases):
        cid = case["id"]
        cpath = case_dir / f"case_{ordinal:03d}.json"
        if not cpath.exists():
            continue
        if term is not None and sha256_bytes(cpath.read_bytes()) != term["rows"][cid]:
            raise RuntimeError(f"Digest mismatch on {cid}")
        rec = json.loads(cpath.read_bytes())
        g = gold_map[cid]

        if rec["vanilla"]["latent_qa"]["visual_forwards"] != 0 or rec["lave"]["latent_qa"]["visual_forwards"] != 0:
            raise RuntimeError(f"Non-zero visual_forwards in latent_qa on {cid}")

        ok = {}
        for arm in ARMS_S8:
            pred = rec["answers"].get(arm)
            if pred is None:
                invalid_ans[arm] += 1
            ok[arm] = (pred == g)

        row = dict(cid=cid, cat=case["question_category"], ok=ok)
        items.append(row)
        by_cat[case["question_category"]].append(row)
        if spec["adapter"] == "mme":
            by_bucket[case["duration_bucket"]].append(row)
        elif spec["adapter"] == "lvb":
            by_bucket[str(case["duration_group"])].append(row)
        if rec["lave"]["reroute_count"] > 0:
            rerouted_items.append(row)

        stats["v_steps"] += rec["vanilla"]["num_steps"]
        stats["v_vcalls"] += rec["vanilla"]["visual_tool_calls"]
        stats["v_frames"] += rec["vanilla"]["total_frames"]
        stats["v_invalid_tools"] += sum(1 for st in rec["vanilla"]["steps"] if st["tool"] == "invalid")
        stats["l_steps"] += rec["lave"]["num_steps"]
        stats["l_vcalls"] += rec["lave"]["visual_tool_calls"]
        stats["l_frames"] += rec["lave"]["total_frames"]
        stats["l_invalid_tools"] += sum(1 for st in rec["lave"]["steps"] if st["tool"] == "invalid")
        stats["control_scans"] += rec["lave"]["control_scans"]
        stats["control_approvals"] += rec["lave"]["control_approvals"]
        stats["reroute_steps"] += rec["lave"]["reroute_count"]
        stats["rx_seconds"] += rec["lave"]["receiver_seconds"]
        stats["elapsed_seconds"] += rec["elapsed_seconds"]

    n = len(items)
    result = dict(
        status=f"qwen25vl_{suite}_s{max_steps}_scored" if n == len(cases) else f"qwen25vl_{suite}_s{max_steps}_partial",
        suite=suite,
        model="Qwen2.5-VL-7B-Instruct",
        max_steps=int(max_steps),
        n=n,
        expected_n=len(cases),
        invalid_answers=dict(invalid_ans),
        efficiency=dict(
            vanilla_mean_steps=round(stats["v_steps"] / max(1, n), 2),
            vanilla_mean_visual_calls=round(stats["v_vcalls"] / max(1, n), 2),
            vanilla_mean_frames=round(stats["v_frames"] / max(1, n), 2),
            vanilla_invalid_tool_steps=stats["v_invalid_tools"],
            lave_mean_steps=round(stats["l_steps"] / max(1, n), 2),
            lave_mean_visual_calls=round(stats["l_vcalls"] / max(1, n), 2),
            lave_mean_frames=round(stats["l_frames"] / max(1, n), 2),
            lave_invalid_tool_steps=stats["l_invalid_tools"],
            control_scans=stats["control_scans"],
            control_approvals=stats["control_approvals"],
            control_pass_rate=round(stats["control_approvals"] / max(1, stats["control_scans"]), 4),
            rerouted_cases=len(rerouted_items),
            total_reroute_steps=stats["reroute_steps"],
            mean_receiver_seconds=round(stats["rx_seconds"] / max(1, n), 2),
            mean_case_seconds=round(stats["elapsed_seconds"] / max(1, n), 2),
        ),
        overall=summarize_s8(items),
        rerouted_subset=summarize_s8(rerouted_items),
        by_bucket={k: summarize_s8(v) for k, v in sorted(by_bucket.items())},
        by_category={k: summarize_s8(v) for k, v in sorted(by_cat.items())},
    )
    out_path = ROOT / f"score_qwen25vl_{suite}_s{max_steps}.json"
    sha = write_json_canonical(out_path, result)
    print(json.dumps(dict(score_sha256=sha, **result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    suite = sys.argv[1]
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    chk = (sys.argv[3] != "0") if len(sys.argv) > 3 else True
    score_suite(suite, max_steps=ms, check_gpu_release=chk)
