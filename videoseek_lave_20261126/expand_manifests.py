"""Append-only blind manifest expander (N >= 300 per benchmark, zero overlap, exact frozen prefixes).

Creates:
  1. preflight_holmes315.json : [0..139] exact preflight_eval140 + [140..314] 175 new unique videos (45/cat x 7)
  2. preflight_mme306.json    : [0..215] exact preflight_mme     + [216..305] 90 new unique videos (102S/102M/102L)
  3. preflight_lvb306.json    : [0..149] exact preflight_lvb150  + [150..305] 156 new long (>=600s) questions (17 cats)

Strictly blind (labels=0), CPU-only.
"""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random

from .holmes_inputs import (
    HOLMES_CATEGORIES,
    HOLMES_JSON,
    HOLMES_VIDEOS,
    sha256_bytes,
    write_json_canonical,
)
from .mme_inputs import MME_DURATIONS, MME_PARQUET, MME_VIDEOS, _probe_video_cv2, parse_mme_options
from .score_lvb import LVB_PARQUET


def expand_holmes_315(eval140_path, pilot14_path, out_path, *, seed=20261127):
    base = json.loads(Path(eval140_path).read_bytes())
    pilot = json.loads(Path(pilot14_path).read_bytes())
    assert base["n"] == 140 and base["labels"] == 0

    used_qids = set(base["order"]) | set(pilot["order"])
    used_vids = {c["video_id"] for c in base["cases"]} | {c["video_id"] for c in pilot["cases"]}

    raw = json.loads(HOLMES_JSON.read_text())
    vmap = {p.stem: p for p in HOLMES_VIDEOS.glob("*.mp4")}

    by_cat_vid = defaultdict(lambda: defaultdict(list))
    for item in raw:
        vid = str(item["video ID"])
        cat = str(item["Question Type"])
        qid = f"{vid}__{item['Question ID']}"
        if vid in vmap and cat in HOLMES_CATEGORIES and qid not in used_qids and vid not in used_vids:
            by_cat_vid[cat][vid].append(item)

    rng = random.Random(seed)
    add_per_cat = 25  # 140 + 7*25 = 315 total unique videos (45 per cat)
    added_by_cat = {}
    chosen_new_vids = set()

    # Pick rarest categories first so every single appended case is guaranteed a brand-new unique video!
    cat_order = sorted(HOLMES_CATEGORIES, key=lambda c: len(by_cat_vid[c]))
    for cat in cat_order:
        vids = [v for v in by_cat_vid[cat].keys() if v not in chosen_new_vids]
        rng.shuffle(vids)
        picked = []
        for v in vids:
            if len(picked) >= add_per_cat:
                break
            cands = list(by_cat_vid[cat][v])
            rng.shuffle(cands)
            picked.append(cands[0])
            chosen_new_vids.add(v)
        if len(picked) != add_per_cat:
            raise RuntimeError(f"Holmes expand {cat}: needed {add_per_cat} unique videos, got {len(picked)}")
        added_by_cat[cat] = picked

    # Round-robin across categories so every prefix slice stays perfectly category-balanced!
    extra_cases = []
    for idx in range(add_per_cat):
        for cat in HOLMES_CATEGORIES:
            item = added_by_cat[cat][idx]
            vid = str(item["video ID"])
            vpath = vmap[vid]
            vmeta = _probe_video_cv2(vpath)
            cid = f"{vid}__{item['Question ID']}"
            extra_cases.append(
                dict(
                    id=cid,
                    video_id=vid,
                    question_id=int(item["Question ID"]),
                    question_category=cat,
                    video_path=str(vpath),
                    duration=vmeta["duration"],
                    fps=vmeta["fps"],
                    question=str(item["Question"]),
                    options={str(k): str(v) for k, v in sorted(item["Options"].items())},
                )
            )

    all_cases = list(base["cases"]) + extra_cases
    assert len(all_cases) == 315
    assert len({c["id"] for c in all_cases}) == 315
    assert len({c["video_id"] for c in all_cases}) == 315
    assert [c["id"] for c in all_cases[:140]] == base["order"]

    manifest = dict(
        status="holmes_eval315_preflight_frozen",
        dataset="Video-Holmes",
        split="eval315",
        base_pin=sha256_bytes(Path(eval140_path).read_bytes()),
        base_n=140,
        labels=0,
        n=len(all_cases),
        unique_videos=len({c["video_id"] for c in all_cases}),
        category_counts=dict(sorted(Counter(c["question_category"] for c in all_cases).items())),
        order=[c["id"] for c in all_cases],
        cases=all_cases,
    )
    sha = write_json_canonical(Path(out_path), manifest)
    return sha, manifest


def expand_mme_306(mme216_path, out_path, *, seed=20261127):
    import pyarrow.parquet as pq

    base = json.loads(Path(mme216_path).read_bytes())
    assert base["n"] == 216 and base["labels"] == 0

    used_qids = set(base["order"])
    used_vids = {c["video_id"] for c in base["cases"]}

    rows = pq.read_table(str(MME_PARQUET)).to_pylist()
    vmap = {p.stem: p for p in MME_VIDEOS.glob("*.mp4")}

    rng = random.Random(seed)
    by_dur_rows = defaultdict(list)
    for r in rows:
        dur = str(r["duration"]).strip().lower()
        vid = str(r["videoID"]).strip()
        qid = str(r["question_id"])
        if dur in MME_DURATIONS and vid in vmap and vid not in used_vids and qid not in used_qids:
            by_dur_rows[dur].append(r)

    add_per_dur = 30  # 72 + 30 = 102 per duration -> 306 total unique videos
    chosen_by_dur = {}
    for dur in MME_DURATIONS:
        pool = list(by_dur_rows[dur])
        rng.shuffle(pool)
        by_task = defaultdict(list)
        for r in pool:
            by_task[str(r["task_type"])].append(r)
        tasks = sorted(by_task.keys())
        local_vids = set()
        picked = []
        while len(picked) < add_per_dur:
            progressed = False
            for t in tasks:
                if len(picked) >= add_per_dur:
                    break
                bucket = by_task[t]
                while bucket:
                    cand = bucket.pop()
                    vid = str(cand["videoID"]).strip()
                    if vid not in local_vids:
                        local_vids.add(vid)
                        picked.append(cand)
                        progressed = True
                        break
            if not progressed:
                break
        if len(picked) != add_per_dur:
            raise RuntimeError(f"MME expand {dur}: needed {add_per_dur}, got {len(picked)}")
        chosen_by_dur[dur] = picked

    extra_cases = []
    for i in range(add_per_dur):
        for j in range(len(MME_DURATIONS)):
            dur = MME_DURATIONS[(i + j) % len(MME_DURATIONS)]
            r = chosen_by_dur[dur][i]
            vid = str(r["videoID"]).strip()
            vpath = vmap[vid]
            vmeta = _probe_video_cv2(vpath)
            extra_cases.append(
                dict(
                    id=str(r["question_id"]),
                    video_id=vid,
                    duration_bucket=dur,
                    domain=str(r["domain"]),
                    sub_category=str(r["sub_category"]),
                    question_category=str(r["task_type"]),
                    video_path=str(vpath),
                    duration=vmeta["duration"],
                    fps=vmeta["fps"],
                    question=str(r["question"]),
                    options=parse_mme_options(r["options"]),
                )
            )

    all_cases = list(base["cases"]) + extra_cases
    assert len(all_cases) == 306
    assert len({c["id"] for c in all_cases}) == 306
    assert len({c["video_id"] for c in all_cases}) == 306
    assert [c["id"] for c in all_cases[:216]] == base["order"]

    manifest = dict(
        status="mme306_preflight_frozen",
        dataset="Video-MME",
        split="mme306",
        base_pin=sha256_bytes(Path(mme216_path).read_bytes()),
        base_n=216,
        labels=0,
        n=len(all_cases),
        unique_videos=len({c["video_id"] for c in all_cases}),
        duration_counts=dict(sorted(Counter(c["duration_bucket"] for c in all_cases).items())),
        task_counts=dict(sorted(Counter(c["question_category"] for c in all_cases).items())),
        domain_counts=dict(sorted(Counter(c["domain"] for c in all_cases).items())),
        order=[c["id"] for c in all_cases],
        cases=all_cases,
    )
    sha = write_json_canonical(Path(out_path), manifest)
    return sha, manifest


def expand_lvb_306(lvb150_path, out_path, *, seed=20261127):
    import pyarrow.parquet as pq

    base = json.loads(Path(lvb150_path).read_bytes())
    assert base["n"] == 150 and base["labels"] == 0

    used_ids = set(base["order"])
    used_vids = {c["video_uid"] for c in base["cases"]}
    lvb_dir = Path(base["cases"][0]["video_path"]).parent
    mp4_map = {p.name: p for p in lvb_dir.glob("*.mp4")}

    rows = pq.read_table(str(LVB_PARQUET)).to_pylist()
    rng = random.Random(seed)
    cands = []
    for r in rows:
        vp = str(r["video_path"])
        cid = str(r["id"])
        dg = int(r["duration_group"])
        if vp not in mp4_map or dg < 600 or cid in used_ids:
            continue
        cands.append(r)

    rng.shuffle(cands)
    by_cat_new = defaultdict(list)
    by_cat_old = defaultdict(list)
    for r in cands:
        cat = str(r["question_category"])
        if str(r["video_id"]) not in used_vids:
            by_cat_new[cat].append(r)
        else:
            by_cat_old[cat].append(r)

    all_cats = sorted(set(by_cat_new.keys()) | set(by_cat_old.keys()))
    add_n = 156  # 150 + 156 = 306 total long-video questions (avg 18/cat across 17 cats)
    picked = []
    seen_new_vids = set()

    while len(picked) < add_n:
        progressed = False
        for cat in all_cats:
            if len(picked) >= add_n:
                break
            b = by_cat_new[cat]
            while b:
                r = b.pop()
                if str(r["video_id"]) not in seen_new_vids:
                    seen_new_vids.add(str(r["video_id"]))
                    picked.append(r)
                    progressed = True
                    break
                else:
                    by_cat_old[cat].append(r)
        if not progressed:
            break

    while len(picked) < add_n:
        progressed = False
        for cat in all_cats:
            if len(picked) >= add_n:
                break
            b = by_cat_old[cat]
            if b:
                picked.append(b.pop())
                progressed = True
        if not progressed:
            break

    if len(picked) != add_n:
        raise RuntimeError(f"LVB expand: needed {add_n}, got {len(picked)}")

    dur_cache = {}
    extra_cases = []
    for idx, r in enumerate(picked):
        vp = str(r["video_path"])
        fpath = mp4_map[vp]
        if vp not in dur_cache:
            dur_cache[vp] = _probe_video_cv2(fpath)
        vmeta = dur_cache[vp]
        choices = [str(r[f"option{i}"]) for i in range(5) if r.get(f"option{i}") not in (None, "", "N/A")]
        extra_cases.append(
            dict(
                id=str(r["id"]),
                video_uid=str(r["video_id"]),
                video_name=vp,
                video_path=str(fpath),
                video_bytes=fpath.stat().st_size,
                duration=vmeta["duration"],
                duration_annotated=float(r.get("duration") or vmeta["duration"]),
                duration_group=int(r["duration_group"]),
                question_category=str(r["question_category"]),
                topic_category=str(r.get("topic_category", "")),
                question=str(r["question"]),
                choices=choices,
                ordinal=150 + idx,
                split="lvb306_expanded",
            )
        )

    all_cases = list(base["cases"]) + extra_cases
    assert len(all_cases) == 306
    assert len({c["id"] for c in all_cases}) == 306
    assert [c["id"] for c in all_cases[:150]] == base["order"]

    manifest = dict(
        status="lvb306_vllm_preflight_frozen",
        dataset="LongVideoBench",
        split="lvb306",
        base_pin=sha256_bytes(Path(lvb150_path).read_bytes()),
        base_n=150,
        labels=0,
        n=len(all_cases),
        unique_videos=len({c["video_uid"] for c in all_cases}),
        duration_groups=dict(sorted(Counter(str(c["duration_group"]) for c in all_cases).items())),
        category_counts=dict(sorted(Counter(c["question_category"] for c in all_cases).items())),
        order=[c["id"] for c in all_cases],
        cases=all_cases,
    )
    sha = write_json_canonical(Path(out_path), manifest)
    return sha, manifest


def main():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run manifest expansion on CPU with CUDA_VISIBLE_DEVICES=''")
    root = Path(".")
    h_sha, h_m = expand_holmes_315(
        root / "preflight_eval140.json",
        Path("/data1/llj/lljmas/qwen35_holmes_videoseek_lave_20261126_01/preflight_pilot14.json"),
        root / "preflight_holmes315.json",
    )
    m_sha, m_m = expand_mme_306(
        root / "preflight_mme.json",
        root / "preflight_mme306.json",
    )
    l_sha, l_m = expand_lvb_306(
        root / "preflight_lvb150.json",
        root / "preflight_lvb306.json",
    )
    print(
        json.dumps(
            dict(
                holmes315=dict(pin=h_sha, n=h_m["n"], unique_videos=h_m["unique_videos"], cats=h_m["category_counts"]),
                mme306=dict(pin=m_sha, n=m_m["n"], unique_videos=m_m["unique_videos"], durs=m_m["duration_counts"]),
                lvb306=dict(pin=l_sha, n=l_m["n"], unique_videos=l_m["unique_videos"], dgs=l_m["duration_groups"], n_cats=len(l_m["category_counts"])),
            ),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
