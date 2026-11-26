"""Single-unseal CPU scorer for Video-Holmes multi-step VideoSeek vs VideoSeek+LAVE."""

from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from .holmes_inputs import HOLMES_JSON, sha256_bytes, write_json_canonical

ROOT = Path(".")
ACTION = ROOT / "action"
ARMS = ("vanilla_text", "vanilla_latent_qa", "lave_route_text", "lave_full")
PAIRS = (
    ("lave_full", "vanilla_text"),
    ("lave_full", "vanilla_latent_qa"),
    ("lave_route_text", "vanilla_text"),
    ("vanilla_latent_qa", "vanilla_text"),
    ("lave_full", "lave_route_text"),
)


def exact_mcnemar(wins, losses):
    m = wins + losses
    if m == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(m, i) for i in range(k + 1))
    return min(1.0, 2.0 * tail / (2**m))


def _gpu_released(pid):
    pid = str(pid)
    proc = Path("/proc") / pid / "stat"
    alive = proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[0] != "Z"
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).split()
    return (not alive) and (pid not in apps)


def summarize_subset(items):
    n = len(items)
    acc = {}
    for arm in ARMS:
        c = sum(1 for row in items if row["ok"][arm])
        acc[arm] = dict(correct=c, n=n, pct=round(100.0 * c / n, 2) if n else 0.0)
    paired = {}
    for a, b in PAIRS:
        w = sum(1 for row in items if row["ok"][a] and not row["ok"][b])
        l = sum(1 for row in items if row["ok"][b] and not row["ok"][a])
        paired[f"{a}_vs_{b}"] = dict(
            win=w,
            loss=l,
            net=w - l,
            delta_pp=round(100.0 * (w - l) / n, 2) if n else 0.0,
            p_two_sided_exact=round(exact_mcnemar(w, l), 6),
        )
    return dict(n=n, accuracy=acc, paired=paired)


def score_split(split="pilot14", check_gpu_release=True):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Score on CPU with CUDA_VISIBLE_DEVICES=''")
    pfile = ROOT / f"preflight_{split}.json"
    pin = sha256_bytes(pfile.read_bytes())
    manifest = json.loads(pfile.read_bytes())

    tpath = ACTION / split / "terminal.json"
    term = json.loads(tpath.read_bytes())
    if term["status"] != "complete" or term["pin"] != pin or term["labels"] != 0:
        raise RuntimeError("Terminal receipt invalid or pin mismatch")
    gpu_rel = _gpu_released(term["pid"]) if check_gpu_release else True
    if check_gpu_release and not gpu_rel:
        raise RuntimeError(f"GPU process {term['pid']} is still active")

    raw_gt = json.loads(HOLMES_JSON.read_text())
    gold_map = {f"{x['video ID']}__{x['Question ID']}": str(x["Answer"]).strip().upper() for x in raw_gt}

    items = []
    by_cat = defaultdict(list)
    rerouted_items = []
    stats = Counter()
    invalid_ans = Counter()

    for ordinal, case in enumerate(manifest["cases"]):
        cid = case["id"]
        cpath = ACTION / split / f"case_{ordinal:03d}.json"
        if sha256_bytes(cpath.read_bytes()) != term["rows"][cid]:
            raise RuntimeError(f"Digest mismatch for {cid}")
        rec = json.loads(cpath.read_bytes())
        g = gold_map[cid]

        # Enforce zero vision calls in Receiver & Latent QA
        if rec["vanilla"]["latent_qa"]["visual_forwards"] != 0 or rec["lave"]["latent_qa"]["visual_forwards"] != 0:
            raise RuntimeError(f"Non-zero visual_forwards in latent_qa on {cid}")

        ok = {}
        for arm in ARMS:
            pred = rec["answers"].get(arm)
            if pred is None:
                invalid_ans[arm] += 1
            ok[arm] = (pred == g)

        row = dict(cid=cid, cat=case["question_category"], ok=ok)
        items.append(row)
        by_cat[case["question_category"]].append(row)
        if rec["lave"]["reroute_count"] > 0:
            rerouted_items.append(row)

        stats["v_steps"] += rec["vanilla"]["num_steps"]
        stats["v_visual_calls"] += rec["vanilla"]["visual_tool_calls"]
        stats["v_frames"] += rec["vanilla"]["total_frames"]
        stats["l_steps"] += rec["lave"]["num_steps"]
        stats["l_visual_calls"] += rec["lave"]["visual_tool_calls"]
        stats["l_frames"] += rec["lave"]["total_frames"]
        stats["control_scans"] += rec["lave"]["control_scans"]
        stats["control_approvals"] += rec["lave"]["control_approvals"]
        stats["reroute_steps"] += rec["lave"]["reroute_count"]
        stats["rx_seconds"] += rec["lave"]["receiver_seconds"]
        stats["elapsed_seconds"] += rec["elapsed_seconds"]

    n = len(items)
    result = dict(
        status=f"holmes_{split}_scored",
        dataset="Video-Holmes",
        split=split,
        n=n,
        unique_videos=manifest["unique_videos"],
        gpu_released=gpu_rel,
        terminal_sha256=sha256_bytes(tpath.read_bytes()),
        invalid_answers=dict(invalid_ans),
        efficiency=dict(
            vanilla_mean_steps=round(stats["v_steps"] / n, 2),
            vanilla_mean_visual_calls=round(stats["v_visual_calls"] / n, 2),
            vanilla_mean_frames=round(stats["v_frames"] / n, 2),
            lave_mean_steps=round(stats["l_steps"] / n, 2),
            lave_mean_visual_calls=round(stats["l_visual_calls"] / n, 2),
            lave_mean_frames=round(stats["l_frames"] / n, 2),
            control_scans=stats["control_scans"],
            control_approvals=stats["control_approvals"],
            control_pass_rate=round(stats["control_approvals"] / max(1, stats["control_scans"]), 4),
            rerouted_cases=len(rerouted_items),
            total_reroute_steps=stats["reroute_steps"],
            mean_receiver_seconds=round(stats["rx_seconds"] / n, 2),
            mean_case_seconds=round(stats["elapsed_seconds"] / n, 2),
        ),
        overall=summarize_subset(items),
        rerouted_subset=summarize_subset(rerouted_items),
        by_category={cat: summarize_subset(sub) for cat, sub in sorted(by_cat.items())},
    )
    sha = write_json_canonical(ROOT / f"score_{split}.json", result)
    print(json.dumps(dict(score_sha256=sha, **result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    score_split(sys.argv[1] if len(sys.argv) > 1 else "pilot14")
