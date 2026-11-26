"""Single-unsealing scorer for the 150 category-balanced LongVideoBench vLLM-LAVE run."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys

from missing_fact_routing_20261124.score import ARMS, paired

from .lvb_inputs import LVB_PARQUET, N_SAMPLES
from .lvb_run import ACTION, PREFLIGHT, ROOT, digest, need, write


ROUTES = (("state_split", "text"), ("state_split", "clock_split"), ("clock_split", "text"))
ENDPOINTS = (
    "focus1_native",
    "focus12_on_policy",
    "focus12_equal_budget",
    "direct_visual_qa_f1",
    "direct_visual_qa_f2_op",
    "direct_visual_qa_f2_eq",
)


def _gpu_released(pid):
    pid = str(pid)
    proc = Path("/proc") / pid / "stat"
    alive = proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).split()
    return (not alive) and (pid not in apps)


def _summarize(items):
    correct = {
        arm: {ep: sum(ans[arm, ep] for ans, _ in items) for ep in ENDPOINTS}
        for arm in ARMS
    }
    routing = {
        ep: {
            f"{a}_vs_{b}": dict(
                paired(items, (a, ep), (b, ep)),
                distinct_f1_cases=sum(r[a] != r[b] for _, r in items),
            )
            for a, b in ROUTES
        }
        for ep in ENDPOINTS
    }
    return dict(n=len(items), correct=correct, routing=routing)


def score_lvb(num_shards=2):
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Score on CPU with hidden GPUs")
    manifest = json.loads(PREFLIGHT.read_bytes())
    pin = digest(PREFLIGHT)
    need(manifest["status"] == "lvb150_vllm_preflight_frozen" and manifest["n"] == N_SAMPLES, "Invalid preflight")

    all_rows = {}
    shard_summaries = []
    for sid in range(num_shards):
        tpath = ACTION / f"shard_{sid}.terminal.json"
        term = json.loads(tpath.read_bytes())
        need(term["status"] == "complete" and term["pin"] == pin and term["labels"] == 0,
             f"Shard {sid} incomplete or mismatched")
        need(_gpu_released(term["pid"]), f"Shard {sid} GPU process still alive")
        all_rows.update(term["rows"])
        shard_summaries.append(dict(shard=sid, cases=term["cases"], gpu_idx=term["gpu_idx"], terminal_sha256=digest(tpath)))
    need(len(all_rows) == N_SAMPLES and set(all_rows) == set(manifest["order"]), "Incomplete 150-case union")

    # Single unsealing of correct_choice strictly for the 150 frozen IDs
    import pyarrow.parquet as pq
    tbl = pq.read_table(LVB_PARQUET, columns=["id", "correct_choice"]).to_pydict()
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    gold_all = {str(cid): letters[int(ch)] for cid, ch in zip(tbl["id"], tbl["correct_choice"])}

    strata = {k: [] for k in ("all", "legal", "f1_triggered", "f2_triggered")}
    by_cat = defaultdict(list)
    by_dur = defaultdict(list)
    k_dist = Counter()
    ctrl_stats = Counter()
    flip_stats = Counter()
    timing_sums = Counter()

    for ordinal, case in enumerate(manifest["cases"]):
        cid = case["id"]
        cpath = ACTION / f"case_{ordinal:03d}" / "case.json"
        need(digest(cpath) == all_rows[cid], f"Case digest mismatch: {cid}")
        rec = json.loads(cpath.read_bytes())
        g = gold_all[cid]

        legal = rec["text_action"] is not None
        trace = rec.get("route_audit")
        addrs = (trace or {}).get("addresses") or []
        k_dist[len(addrs)] += 1
        if trace is not None:
            ctrl_stats["scanned"] += 1
            if trace.get("decision") is None:
                ctrl_stats["no_nomination"] += 1
            elif addrs:
                ctrl_stats["approved"] += 1
            else:
                ctrl_stats["rejected_by_controls"] += 1

        routes_f1 = {a: tuple(rec["routes"]["indices"][a]) for a in ARMS}
        f1_trig = len(set(routes_f1.values())) > 1
        f2_trig = bool(rec["focus2_plan"]["triggered"])

        if routes_f1["state_split"] != routes_f1["text"]:
            flip_stats["distinct_st"] += 1
            a1_s = (rec["qa1"]["state_split"]["qa"] or {}).get("answer")
            a1_t = (rec["qa1"]["text"]["qa"] or {}).get("answer")
            a2_s = rec["qa2"]["state_split"]["on_policy_answer"]
            a2_t = rec["qa2"]["text"]["on_policy_answer"]
            dv1_s = rec["direct_visual_qa"]["focus1"]["state_split"]
            dv1_t = rec["direct_visual_qa"]["focus1"]["text"]
            flip_stats["qa1_diff_letter"] += int(a1_s != a1_t)
            flip_stats["qa2_diff_letter"] += int(a2_s != a2_t)
            flip_stats["dvqa1_diff_letter"] += int(dv1_s != dv1_t)

        ans = {}
        for arm in ARMS:
            a1 = (rec["qa1"][arm]["qa"] or {}).get("answer")
            a2_op = rec["qa2"][arm]["on_policy_answer"]
            a2_eq = rec["qa2"][arm]["equal_budget_answer"]
            dv1 = rec["direct_visual_qa"]["focus1"][arm]
            dv2_op = rec["direct_visual_qa"]["focus2"][arm]["on_policy"]
            dv2_eq = rec["direct_visual_qa"]["focus2"][arm]["equal_budget"]
            ans[arm, "focus1_native"] = (a1 == g)
            ans[arm, "focus12_on_policy"] = (a2_op == g)
            ans[arm, "focus12_equal_budget"] = (a2_eq == g)
            ans[arm, "direct_visual_qa_f1"] = (dv1 == g)
            ans[arm, "direct_visual_qa_f2_op"] = (dv2_op == g)
            ans[arm, "direct_visual_qa_f2_eq"] = (dv2_eq == g)

        item = (ans, routes_f1)
        strata["all"].append(item)
        if legal:
            strata["legal"].append(item)
        if f1_trig:
            strata["f1_triggered"].append(item)
        if f2_trig:
            strata["f2_triggered"].append(item)
        by_cat[case["question_category"]].append(item)
        by_dur[str(case["duration_group"])].append(item)

        for k_t, v_t in rec.get("timings", {}).items():
            timing_sums[k_t] += float(v_t)
        timing_sums["elapsed_seconds"] += float(rec.get("elapsed_seconds", 0.0))

    result = dict(
        status="lvb150_vllm_scored",
        split="lvb_eval",
        n=N_SAMPLES,
        unique_videos=manifest["unique_videos"],
        gpu_released=True,
        shards=shard_summaries,
        k_distribution=dict(sorted(k_dist.items())),
        control_stats=dict(ctrl_stats),
        answer_flip_stats=dict(flip_stats),
        mean_stage_seconds={k: round(v / N_SAMPLES, 2) for k, v in sorted(timing_sums.items())},
        qa={name: _summarize(items) for name, items in strata.items()},
        by_duration_group={k: _summarize(items) for k, items in sorted(by_dur.items())},
        by_category={
            cat: {
                "n": len(items),
                "focus1_native": {a: sum(ans[a, "focus1_native"] for ans, _ in items) for a in ARMS},
                "focus12_on_policy": {a: sum(ans[a, "focus12_on_policy"] for ans, _ in items) for a in ARMS},
                "direct_visual_qa_f1": {a: sum(ans[a, "direct_visual_qa_f1"] for ans, _ in items) for a in ARMS},
                "direct_visual_qa_f2_op": {a: sum(ans[a, "direct_visual_qa_f2_op"] for ans, _ in items) for a in ARMS},
            }
            for cat, items in sorted(by_cat.items())
        },
    )
    sha = write(ROOT / "score.json", result)
    print(json.dumps(dict(score_sha256=sha, **result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    num = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    score_lvb(num)
