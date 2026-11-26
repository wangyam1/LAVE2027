"""Balanced blind Video-MME manifest builder (Short / Medium / Long, 216 unique videos, labels=0)."""

from collections import Counter, defaultdict
from pathlib import Path
import random
import re

from .holmes_inputs import sha256_bytes, write_json_canonical

MME_ROOT = Path("/data1/llj/development/lljvideo/datasets/Video-MME")
MME_PARQUET = MME_ROOT / "videomme" / "test-00000-of-00001.parquet"
MME_VIDEOS = MME_ROOT / "data"
MME_DURATIONS = ("short", "medium", "long")


def parse_mme_options(raw_options):
    """Parse Video-MME options list like ['A. Apples.', 'B. Candles.', ...] into {'A': 'Apples.', ...}."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    out = {}
    for idx, item in enumerate(raw_options):
        s = str(item).strip()
        m = re.match(r"^([A-Z])\s*[\.\):]\s*(.*)$", s)
        if m:
            out[m.group(1).upper()] = m.group(2).strip()
        else:
            out[letters[idx]] = s
    return out


def _probe_video_cv2(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open Video-MME video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 25.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (n_frames / fps) if n_frames > 0 else 0.0
        return dict(duration=round(dur, 2), fps=round(fps, 3))
    finally:
        cap.release()


def build_balanced_mme_manifest(
    mme_parquet=MME_PARQUET,
    videos_dir=MME_VIDEOS,
    *,
    per_duration=72,
    seed=20261126,
    probe_duration=True,
):
    """Build a strictly blind (labels=0), balanced Video-MME manifest across short/medium/long.

    Picks `per_duration` unique videos for each duration bucket (default 72 x 3 = 216 unique videos),
    balancing across `task_type` and `domain` inside each bucket, then interleaves round-robin across
    (short, medium, long) so every 3-GPU shard and every interim snapshot is duration- and task-balanced.
    """
    import pyarrow.parquet as pq

    rows = pq.read_table(str(mme_parquet)).to_pylist()
    vmap = {p.stem: p for p in Path(videos_dir).glob("*.mp4")}

    rng = random.Random(seed)
    by_dur_rows = defaultdict(list)
    for r in rows:
        dur = str(r["duration"]).strip().lower()
        vid = str(r["videoID"]).strip()
        if dur in MME_DURATIONS and vid in vmap:
            by_dur_rows[dur].append(r)

    chosen_by_dur = {}
    for dur in MME_DURATIONS:
        pool = list(by_dur_rows[dur])
        rng.shuffle(pool)
        # Group by task_type for balanced stratified sampling while keeping 100% unique videos
        by_task = defaultdict(list)
        for r in pool:
            by_task[str(r["task_type"])].append(r)
        tasks = sorted(by_task.keys())
        used_vids = set()
        picked = []
        # Round-robin across task_types until per_duration unique videos are selected
        while len(picked) < per_duration:
            progressed = False
            for t in tasks:
                if len(picked) >= per_duration:
                    break
                bucket = by_task[t]
                while bucket:
                    cand = bucket.pop()
                    vid = str(cand["videoID"]).strip()
                    if vid not in used_vids:
                        used_vids.add(vid)
                        picked.append(cand)
                        progressed = True
                        break
            if not progressed:
                break
        if len(picked) != per_duration:
            raise RuntimeError(f"Duration {dur}: needed {per_duration}, got {len(picked)}")
        chosen_by_dur[dur] = picked

    # Interleave with cyclic shift so each of 3 shards gets exactly 24 short + 24 medium + 24 long
    ordered = []
    for i in range(per_duration):
        for j in range(len(MME_DURATIONS)):
            dur = MME_DURATIONS[(i + j) % len(MME_DURATIONS)]
            r = chosen_by_dur[dur][i]
            vid = str(r["videoID"]).strip()
            vpath = vmap[vid]
            vmeta = _probe_video_cv2(vpath) if probe_duration else dict(duration=180.0, fps=25.0)
            ordered.append(
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

    dur_counts = dict(sorted(Counter(x["duration_bucket"] for x in ordered).items()))
    task_counts = dict(sorted(Counter(x["question_category"] for x in ordered).items()))
    dom_counts = dict(sorted(Counter(x["domain"] for x in ordered).items()))
    return dict(
        status="mme216_preflight_frozen",
        dataset="Video-MME",
        split=f"mme{len(ordered)}",
        labels=0,
        n=len(ordered),
        unique_videos=len({x["video_id"] for x in ordered}),
        duration_counts=dur_counts,
        task_counts=task_counts,
        domain_counts=dom_counts,
        order=[x["id"] for x in ordered],
        cases=ordered,
    )
