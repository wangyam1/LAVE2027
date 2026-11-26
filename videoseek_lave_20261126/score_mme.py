"""Single-unseal CPU scorer for multi-GPU Video-MME VideoSeek vs VideoSeek+LAVE."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from .holmes_inputs import sha256_bytes, write_json_canonical
from .mme_inputs import MME_PARQUET
from .score_holmes import ARMS, _gpu_released, summarize_subset

ROOT = Path(".")
ACTION = ROOT / "action" / "mme"


def score_mme(num_shards=3, check_gpu_release=True):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Score on CPU with CUDA_VISIBLE_DEVICES=''")
    pfile = ROOT / "preflight_mme.json"
    pin = sha256_bytes(pfile.read_bytes())
    manifest = json.loads(pfile.read_bytes())

    all_rows = {}
    shard_meta = []
    for sid in range(int(num_shards)):
        tpath = ACTION / f"shard_{sid}.terminal.json"
        term = json.loads(tpath.read_bytes())
        if term["status"] != "complete" or term["pin"] != pin or term["labels"] != 0:
            raise RuntimeError(f"Shard {sid} terminal invalid or pin mismatch")
        if check_gpu_release and not _gpu_released(term["pid"]):
            raise RuntimeError(f"Shard {sid} GPU process {term['pid']} still alive")
        all_rows.update(term["rows"])
        shard_meta.append(
            dict(shard=sid, gpu_idx=term["gpu_idx"], cases=term["cases"], terminal_sha256=sha256_bytes(tpath.read_bytes()))
        )
    if len(all_rows) != manifest["n"] or set(all_rows) != set(manifest["order"]):
        raise RuntimeError(f"Incomplete shard union: {len(all_rows)} != {manifest['n']}")

    import pyarrow.parquet as pq

    tbl = pq.read_table(str(MME_PARQUET), columns=["question_id", "answer"]).to_pydict()
    gold_map = {str(qid): str(ans).strip().upper() for qid, ans in zip(tbl["question_id"], tbl["answer"])}

    items = []
    rerouted_items = []
    by_dur = defaultdict(list)
    by_task = defaultdict(list)
    by_dom = defaultdict(list)
    stats = Counter()
    invalid_ans = Counter()

    for ordinal, case in enumerate(manifest["cases"]):
        cid = case["id"]
        cpath = ACTION / f"case_{ordinal:03d}.json"
        if sha256_bytes(cpath.read_bytes()) != all_rows[cid]:
            raise RuntimeError(f"Digest mismatch on {cid}")
        rec = json.loads(cpath.read_bytes())
        g = gold_map[cid]

        if rec["vanilla"]["latent_qa"]["visual_forwards"] != 0 or rec["lave"]["latent_qa"]["visual_forwards"] != 0:
            raise RuntimeError(f"Non-zero visual_forwards in latent_qa on {cid}")

        ok = {}
        for arm in ARMS:
            pred = rec["answers"].get(arm)
            if pred is None:
                invalid_ans[arm] += 1
            ok[arm] = (pred == g)

        row = dict(cid=cid, dur=case["duration_bucket"], task=case["question_category"], ok=ok)
        items.append(row)
        by_dur[case["duration_bucket"]].append(row)
        by_task[case["question_category"]].append(row)
        by_dom[case["domain"]].append(row)
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
        status="mme_videoseek_lave_scored",
        dataset="Video-MME",
        split=manifest["split"],
        n=n,
        unique_videos=manifest["unique_videos"],
        gpu_released=True,
        shards=shard_meta,
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
        by_duration={k: summarize_subset(v) for k, v in sorted(by_dur.items())},
        by_domain={k: summarize_subset(v) for k, v in sorted(by_dom.items())},
        by_task_type={k: summarize_subset(v) for k, v in sorted(by_task.items())},
    )
    sha = write_json_canonical(ROOT / "score_mme.json", result)
    print(json.dumps(dict(score_sha256=sha, **result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    score_mme(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
