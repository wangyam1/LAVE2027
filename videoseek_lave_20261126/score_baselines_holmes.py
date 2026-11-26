"""Single-unseal CPU scorer joining Clock-Split & VideoSeek-Image baselines with sealed Video-Holmes eval140."""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from .holmes_inputs import HOLMES_JSON, sha256_bytes, write_json_canonical
from .score_holmes import _gpu_released, exact_mcnemar

ROOT = Path(".")
ACTION = ROOT / "action" / "baselines_eval140"
SEALED_EVAL140 = Path("/data1/llj/lljmas/qwen35_holmes_videoseek_lave_20261126_01/action/eval140")

ALL_ARMS = (
    "vanilla_text",
    "vanilla_step4",
    "vanilla_latent_qa",
    "clock_split_text",
    "clock_split_step4",
    "clock_split_uniform_qa",
    "videoseek_image",
    "videoseek_image_step4",
    "lave_route_text",
    "lave_step4",
    "lave_full",
)

KEY_PAIRS = (
    ("lave_route_text", "clock_split_text"),
    ("lave_route_text", "videoseek_image"),
    ("lave_full", "videoseek_image"),
    ("lave_full", "clock_split_uniform_qa"),
    ("lave_route_text", "vanilla_text"),
    ("videoseek_image", "vanilla_text"),
    ("clock_split_text", "vanilla_text"),
    ("vanilla_latent_qa", "clock_split_uniform_qa"),
)


def summarize(items):
    n = len(items)
    acc = {}
    for arm in ALL_ARMS:
        c = sum(1 for r in items if r["ok"].get(arm, False))
        acc[arm] = dict(correct=c, n=n, pct=round(100.0 * c / n, 2) if n else 0.0)
    paired = {}
    for a, b in KEY_PAIRS:
        w = sum(1 for r in items if r["ok"].get(a, False) and not r["ok"].get(b, False))
        l = sum(1 for r in items if r["ok"].get(b, False) and not r["ok"].get(a, False))
        paired[f"{a}_vs_{b}"] = dict(
            win=w,
            loss=l,
            net=w - l,
            delta_pp=round(100.0 * (w - l) / n, 2) if n else 0.0,
            p_two_sided_exact=round(exact_mcnemar(w, l), 6),
        )
    return dict(n=n, accuracy=acc, paired=paired)


def score_baselines(num_shards=3, check_gpu_release=True, sealed_dir=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Score on CPU with CUDA_VISIBLE_DEVICES=''")
    pfile = ROOT / "preflight_eval140.json"
    pin = sha256_bytes(pfile.read_bytes())
    manifest = json.loads(pfile.read_bytes())
    sdir = Path(sealed_dir) if sealed_dir else SEALED_EVAL140

    all_rows = {}
    shard_meta = []
    for sid in range(int(num_shards)):
        tpath = ACTION / f"shard_{sid}.terminal.json"
        if not tpath.exists():
            continue
        term = json.loads(tpath.read_bytes())
        if term["status"] != "complete" or term["pin"] != pin or term["labels"] != 0:
            raise RuntimeError(f"Shard {sid} terminal invalid or pin mismatch")
        if check_gpu_release and not _gpu_released(term["pid"]):
            raise RuntimeError(f"Shard {sid} GPU process {term['pid']} still active")
        all_rows.update(term["rows"])
        shard_meta.append(dict(shard=sid, gpu_idx=term["gpu_idx"], cases=term["cases"], terminal_sha256=sha256_bytes(tpath.read_bytes())))

    raw_gt = json.loads(HOLMES_JSON.read_text())
    gold_map = {f"{x['video ID']}__{x['Question ID']}": str(x["Answer"]).strip().upper() for x in raw_gt}

    items = []
    by_cat = defaultdict(list)
    stats = Counter()
    lat_rows = []
    invalid_ans = Counter()

    for ordinal, case in enumerate(manifest["cases"]):
        cid = case["id"]
        cpath = ACTION / f"case_{ordinal:03d}.json"
        if not cpath.exists():
            continue
        if cid in all_rows and sha256_bytes(cpath.read_bytes()) != all_rows[cid]:
            raise RuntimeError(f"Digest mismatch on {cid}")
        rec = json.loads(cpath.read_bytes())
        g = gold_map[cid]

        spath = sdir / f"case_{ordinal:03d}.json"
        srec = json.loads(spath.read_bytes()) if spath.exists() else None
        if srec is not None and srec["cid"] != cid:
            raise RuntimeError(f"Sealed case mismatch on ordinal {ordinal}: {srec['cid']} != {cid}")

        if rec["clock_split"]["uniform_qa"]["visual_forwards"] != 0:
            raise RuntimeError(f"Non-zero visual_forwards in clock_split uniform_qa on {cid}")

        preds = dict(rec["answers"])
        if srec is not None:
            preds.update(srec["answers"])

        ok = {}
        for arm in ALL_ARMS:
            p = preds.get(arm)
            if p is None:
                invalid_ans[arm] += 1
            ok[arm] = (p == g)

        row = dict(cid=cid, cat=case["question_category"], ok=ok)
        items.append(row)
        by_cat[case["question_category"]].append(row)

        cs = rec["clock_split"]
        vi = rec["videoseek_image"]
        stats["clk_steps"] += cs["num_steps"]
        stats["clk_vcalls"] += cs["visual_tool_calls"]
        stats["clk_frames"] += cs["total_frames"]
        stats["clk_reroutes"] += cs["reroute_count"]
        stats["img_steps"] += vi["num_steps"]
        stats["img_vcalls"] += vi["visual_tool_calls"]
        stats["img_vlm_turns"] += vi["vlm_turns"]
        stats["img_frames"] += vi["total_frames"]
        stats["img_frames_fed"] += vi["total_frames_fed_vlm"]
        stats["img_pruned"] += vi["pruned_frames"]
        stats["img_prompt_toks"] += vi["total_prompt_tokens"]
        stats["img_output_toks"] += vi["total_output_tokens"]
        stats["img_seconds"] += vi["elapsed_seconds"]

        if rec.get("latency_audit"):
            la = dict(rec["latency_audit"])
            la["videoseek_image_s"] = vi["elapsed_seconds"]
            la["videoseek_image_frames"] = vi["total_frames"]
            la["videoseek_image_frames_fed"] = vi["total_frames_fed_vlm"]
            lat_rows.append(la)

    n = len(items)
    lat_summary = {}
    if lat_rows:
        m = len(lat_rows)
        for k in (
            "vanilla_text_s",
            "vanilla_latent_qa_s",
            "vanilla_frames",
            "vanilla_visual_calls",
            "lave_route_text_s",
            "lave_full_s",
            "lave_receiver_s",
            "lave_latent_qa_s",
            "lave_frames",
            "lave_visual_calls",
            "videoseek_image_s",
            "videoseek_image_frames",
            "videoseek_image_frames_fed",
        ):
            lat_summary[f"mean_{k}"] = round(sum(r[k] for r in lat_rows) / m, 2)
        lat_summary["n_latency_cases"] = m

    result = dict(
        status="holmes_eval140_baselines_scored" if n == manifest["n"] else "holmes_eval140_baselines_partial",
        dataset="Video-Holmes",
        split="eval140_baselines",
        n=n,
        expected_n=manifest["n"],
        shards=shard_meta,
        invalid_answers=dict(invalid_ans),
        efficiency=dict(
            clock_split_mean_steps=round(stats["clk_steps"] / max(1, n), 2),
            clock_split_mean_visual_calls=round(stats["clk_vcalls"] / max(1, n), 2),
            clock_split_mean_frames=round(stats["clk_frames"] / max(1, n), 2),
            videoseek_image_mean_steps=round(stats["img_steps"] / max(1, n), 2),
            videoseek_image_mean_visual_calls=round(stats["img_vcalls"] / max(1, n), 2),
            videoseek_image_mean_vlm_turns=round(stats["img_vlm_turns"] / max(1, n), 2),
            videoseek_image_mean_frames=round(stats["img_frames"] / max(1, n), 2),
            videoseek_image_mean_frames_fed_vlm=round(stats["img_frames_fed"] / max(1, n), 2),
            videoseek_image_mean_pruned=round(stats["img_pruned"] / max(1, n), 2),
            videoseek_image_mean_prompt_tokens=round(stats["img_prompt_toks"] / max(1, n), 1),
            videoseek_image_mean_output_tokens=round(stats["img_output_toks"] / max(1, n), 1),
            videoseek_image_mean_seconds=round(stats["img_seconds"] / max(1, n), 2),
        ),
        standalone_latency_audit=lat_summary,
        overall=summarize(items),
        by_category={k: summarize(v) for k, v in sorted(by_cat.items())},
    )
    sha = write_json_canonical(ROOT / "score_baselines_eval140.json", result)
    print(json.dumps(dict(score_sha256=sha, **result), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    score_baselines(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
