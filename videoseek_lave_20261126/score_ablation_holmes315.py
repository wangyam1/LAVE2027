"""CPU single-unseal scorer for Stage 3 matched-cohort Table 2 ablations on Video-Holmes-315 (N=315, max_steps=8).

Merges the main holmes315_s8 run (vanilla_text, lave_route_text, vanilla_latent_visonly, lave_full_visonly)
with Stage 3 ablation runs (clock_split, no_calib, no_gate, fixed_k1, lave_msweep, videoseek_image, unshared_latency)
on the exact same N=315 cohort and computes exact two-sided McNemar p-values + stage latency breakdown.
"""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys

from .holmes_inputs import HOLMES_JSON, sha256_bytes, write_json_canonical
from .run_ablation_holmes315 import HOLMES315_PIN
from .score_holmes import exact_mcnemar


def load_holmes_gold():
    raw = json.loads(HOLMES_JSON.read_text())
    return {f"{x['video ID']}__{x['Question ID']}": str(x["Answer"]).strip().upper() for x in raw}


def paired_stat(rows, arm_a, arm_b):
    shared = [r for r in rows if arm_a in r["ok"] and arm_b in r["ok"]]
    n = len(shared)
    if n == 0:
        return None
    w = sum(1 for r in shared if r["ok"][arm_a] and not r["ok"][arm_b])
    l = sum(1 for r in shared if r["ok"][arm_b] and not r["ok"][arm_a])
    return dict(
        n=n,
        win=w,
        loss=l,
        net=w - l,
        delta_pp=round(100.0 * (w - l) / n, 2),
        p_two_sided_exact=round(exact_mcnemar(w, l), 6),
    )


def score_ablations(
    manifest_path="preflight_holmes315.json",
    main_dir="action/holmes315_s8",
    ablation_dir="action/ablation_qwen25vl_7b_s8",
    tag="qwen25vl_7b",
    max_steps=8,
):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Score on CPU with CUDA_VISIBLE_DEVICES=''")

    pfile = Path(manifest_path)
    pin = sha256_bytes(pfile.read_bytes())
    if pin != HOLMES315_PIN:
        raise RuntimeError(f"Manifest PIN mismatch: {pin} != {HOLMES315_PIN}")
    manifest = json.loads(pfile.read_bytes())
    cases = manifest["cases"]
    gold_map = load_holmes_gold()

    main_p = Path(main_dir)
    abl_p = Path(ablation_dir)
    traj_p = abl_p / "traj_bundle"
    img_p = abl_p / "videoseek_image"
    lat_p = abl_p / "unshared_latency"

    rows = []
    invalid_counts = Counter()
    eff_sums = defaultdict(Counter)
    eff_counts = Counter()
    all_arms = set()

    for ordinal, case in enumerate(cases):
        cid = case["id"]
        g = gold_map[cid]
        ok = {}

        # Main run arms (Vanilla + Production LAVE)
        mc = main_p / f"case_{ordinal:03d}.json"
        if mc.exists():
            mrec = json.loads(mc.read_bytes())
            for arm, pred in (mrec.get("answers") or {}).items():
                all_arms.add(arm)
                if pred is None:
                    invalid_counts[arm] += 1
                ok[arm] = (pred == g)
            for key in ("vanilla", "lave"):
                tr = mrec.get(key) or {}
                if tr:
                    eff_counts[key] += 1
                    eff_sums[key]["steps"] += tr.get("num_steps", 0)
                    eff_sums[key]["vcalls"] += tr.get("visual_tool_calls", 0)
                    eff_sums[key]["frames"] += tr.get("total_frames", 0)
                    eff_sums[key]["rx_s"] += tr.get("receiver_seconds", 0.0)

        # Trajectory ablation bundle
        tc = traj_p / f"case_{ordinal:03d}.json"
        if tc.exists():
            trec = json.loads(tc.read_bytes())
            for arm, pred in (trec.get("answers") or {}).items():
                all_arms.add(arm)
                if pred is None:
                    invalid_counts[arm] += 1
                ok[arm] = (pred == g)
            for sub_name, sub_tr in (trec.get("arms") or {}).items():
                eff_counts[sub_name] += 1
                eff_sums[sub_name]["steps"] += sub_tr.get("num_steps", 0)
                eff_sums[sub_name]["vcalls"] += sub_tr.get("visual_tool_calls", 0)
                eff_sums[sub_name]["frames"] += sub_tr.get("total_frames", 0)
                eff_sums[sub_name]["rx_s"] += sub_tr.get("receiver_seconds", 0.0)
                eff_sums[sub_name]["reroutes"] += sub_tr.get("reroute_count", 0)
                if sub_tr.get("latent_qa") and sub_tr["latent_qa"].get("visual_forwards", 0) != 0:
                    raise RuntimeError(f"Non-zero visual_forwards in {sub_name} on {cid}")
                if sub_tr.get("uniform_qa") and sub_tr["uniform_qa"].get("visual_forwards", 0) != 0:
                    raise RuntimeError(f"Non-zero visual_forwards in clock_split uniform_qa on {cid}")

        # VideoSeek-Image standalone
        ic = img_p / f"case_{ordinal:03d}.json"
        if ic.exists():
            irec = json.loads(ic.read_bytes())
            for arm, pred in (irec.get("answers") or {}).items():
                all_arms.add(arm)
                if pred is None:
                    invalid_counts[arm] += 1
                ok[arm] = (pred == g)
            vi = irec.get("videoseek_image") or {}
            eff_counts["videoseek_image"] += 1
            eff_sums["videoseek_image"]["steps"] += vi.get("num_steps", 0)
            eff_sums["videoseek_image"]["vcalls"] += vi.get("visual_tool_calls", 0)
            eff_sums["videoseek_image"]["frames"] += vi.get("total_frames", 0)
            eff_sums["videoseek_image"]["fed_frames"] += vi.get("total_frames_fed_vlm", 0)
            eff_sums["videoseek_image"]["pruned_frames"] += vi.get("pruned_frames", 0)
            eff_sums["videoseek_image"]["prompt_tokens"] += vi.get("total_prompt_tokens", 0)

        rows.append(dict(cid=cid, cat=case["question_category"], ok=ok))

    arm_summary = {}
    for arm in sorted(all_arms):
        sub = [r for r in rows if arm in r["ok"]]
        c = sum(1 for r in sub if r["ok"][arm])
        n = len(sub)
        arm_summary[arm] = dict(
            correct=c,
            n=n,
            pct=round(100.0 * c / n, 2) if n else 0.0,
            vs_vanilla_text=paired_stat(rows, arm, "vanilla_text") if arm != "vanilla_text" else None,
            vs_lave_full_visonly=paired_stat(rows, arm, "lave_full_visonly") if arm != "lave_full_visonly" else None,
        )

    eff_summary = {}
    for k, cnt in sorted(eff_counts.items()):
        s = eff_sums[k]
        eff_summary[k] = {m: round(v / max(1, cnt), 2) for m, v in sorted(s.items())}
        eff_summary[k]["n"] = cnt

    # Unshared standalone latency audit summary
    lat_summary = {}
    if lat_p.exists():
        lat_files = sorted(lat_p.glob("case_*.json"))
        if lat_files:
            recs = [json.loads(p.read_bytes()) for p in lat_files]
            n_lat = len(recs)
            lat_summary["n"] = n_lat
            for field in (
                "vanilla_text_s",
                "vanilla_latent_qa_s",
                "vanilla_latent_visonly_s",
                "vanilla_frames",
                "vanilla_visual_calls",
                "lave_route_text_s",
                "lave_full_s",
                "lave_full_visonly_s",
                "lave_receiver_s",
                "lave_latent_qa_s",
                "lave_frames",
                "lave_visual_calls",
            ):
                vals = [float(r[field]) for r in recs if field in r]
                if vals:
                    lat_summary[field] = round(sum(vals) / len(vals), 2)
            for st_key in ("vanilla_stages", "lave_stages"):
                stage_acc = defaultdict(list)
                for r in recs:
                    for sk, sv in (r.get(st_key) or {}).items():
                        stage_acc[sk].append(float(sv))
                if stage_acc:
                    lat_summary[st_key] = {sk: round(sum(vs) / len(vs), 2) for sk, vs in sorted(stage_acc.items())}

    out = dict(
        status=f"ablation_{tag}_s{max_steps}_scored",
        tag=tag,
        max_steps=int(max_steps),
        pin=pin,
        cohort_n=len(cases),
        invalid_answers=dict(invalid_counts),
        arms=arm_summary,
        efficiency=eff_summary,
        unshared_latency=lat_summary,
    )
    out_path = Path(f"score_ablation_{tag}_holmes315_s{max_steps}.json")
    sha = write_json_canonical(out_path, out)
    print(json.dumps(dict(score_sha256=sha, **out), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "qwen25vl_7b"
    ms = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    main_d = sys.argv[3] if len(sys.argv) > 3 else f"action/holmes315_s{ms}"
    abl_d = sys.argv[4] if len(sys.argv) > 4 else f"action/ablation_{tag}_s{ms}"
    score_ablations(main_dir=main_d, ablation_dir=abl_d, tag=tag, max_steps=ms)
