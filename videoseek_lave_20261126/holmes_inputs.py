"""Category-balanced blind Video-Holmes manifest builder (pilot=14, eval=140, zero labels inside preflight)."""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random


HOLMES_ROOT = Path("/data1/llj/development/lljvideo/CLAUDEvidoe/datasets/Video-Holmes")
HOLMES_JSON = HOLMES_ROOT / "test_Video-Holmes.json"
HOLMES_VIDEOS = HOLMES_ROOT / "videos_cropped"
HOLMES_CATEGORIES = ("CTI", "IMC", "MHR", "PAR", "SR", "TA", "TCI")


def _probe_video_meta(video_path):
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        duration = float(container.duration) / 1e6 if container.duration else 0.0
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        if duration <= 0.0 and stream.frames and fps > 0:
            duration = float(stream.frames) / fps
        return dict(duration=round(duration, 2), fps=round(fps, 3))


def build_balanced_holmes_manifests(
    holmes_json=HOLMES_JSON,
    videos_dir=HOLMES_VIDEOS,
    *,
    pilot_per_cat=2,
    eval_per_cat=20,
    seed=20261126,
    probe_duration=True,
):
    """Build strictly disjoint blind pilot (14) and eval (140) manifests balanced across all 7 categories.

    Maximizes unique video coverage so neither split is dominated by a few videos.
    Strips Answer and Explanation completely so preflight files remain strictly blind (labels=0).
    """
    raw = json.loads(Path(holmes_json).read_text())
    vmap = {p.stem: p for p in Path(videos_dir).glob("*.mp4")}
    by_cat = defaultdict(list)
    for item in raw:
        vid = str(item["video ID"])
        cat = str(item["Question Type"])
        if vid in vmap and cat in HOLMES_CATEGORIES:
            by_cat[cat].append(item)

    rng = random.Random(seed)
    used_qids = set()
    used_vids_pilot = set()
    used_vids_eval = set()
    pilot_by_cat = {}
    eval_by_cat = {}

    for cat in HOLMES_CATEGORIES:
        pool = list(by_cat[cat])
        rng.shuffle(pool)

        # Group by video to maximize unique video diversity
        by_vid = defaultdict(list)
        for item in pool:
            by_vid[str(item["video ID"])].append(item)
        vids = list(by_vid.keys())
        rng.shuffle(vids)

        def _pick(target_n, preferred_unused_vids):
            chosen = []
            # Pass 1: prefer videos not yet used in this split
            for v in vids:
                if len(chosen) >= target_n:
                    break
                if v in preferred_unused_vids:
                    continue
                for cand in by_vid[v]:
                    qid = f"{cand['video ID']}__{cand['Question ID']}"
                    if qid not in used_qids:
                        used_qids.add(qid)
                        preferred_unused_vids.add(v)
                        chosen.append(cand)
                        break
            # Pass 2: allow reused videos if needed
            for v in vids:
                if len(chosen) >= target_n:
                    break
                for cand in by_vid[v]:
                    if len(chosen) >= target_n:
                        break
                    qid = f"{cand['video ID']}__{cand['Question ID']}"
                    if qid not in used_qids:
                        used_qids.add(qid)
                        preferred_unused_vids.add(v)
                        chosen.append(cand)
            if len(chosen) != target_n:
                raise RuntimeError(f"Category {cat}: needed {target_n}, got {len(chosen)}")
            return chosen

        pilot_by_cat[cat] = _pick(pilot_per_cat, used_vids_pilot)
        eval_by_cat[cat] = _pick(eval_per_cat, used_vids_eval)

    def _interleave_and_strip(cat_dict, split_name):
        # Round-robin across categories so partial progress snapshots are always category-balanced!
        ordered = []
        max_len = max(len(v) for v in cat_dict.values())
        dur_cache = {}
        for idx in range(max_len):
            for cat in HOLMES_CATEGORIES:
                if idx < len(cat_dict[cat]):
                    item = cat_dict[cat][idx]
                    vid = str(item["video ID"])
                    vpath = vmap[vid]
                    if probe_duration:
                        if vid not in dur_cache:
                            dur_cache[vid] = _probe_video_meta(vpath)
                        vmeta = dur_cache[vid]
                    else:
                        vmeta = dict(duration=180.0, fps=25.0)
                    cid = f"{vid}__{item['Question ID']}"
                    ordered.append(
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
        counts = {c: sum(1 for x in ordered if x["question_category"] == c) for c in HOLMES_CATEGORIES}
        manifest = dict(
            status=f"holmes_{split_name}_preflight_frozen",
            dataset="Video-Holmes",
            split=split_name,
            labels=0,
            n=len(ordered),
            unique_videos=len({x["video_id"] for x in ordered}),
            category_counts=counts,
            order=[x["id"] for x in ordered],
            cases=ordered,
        )
        return manifest

    pilot = _interleave_and_strip(pilot_by_cat, "pilot14")
    eval_m = _interleave_and_strip(eval_by_cat, "eval140")
    if set(pilot["order"]) & set(eval_m["order"]):
        raise RuntimeError("Pilot and Eval manifests overlapped!")
    return pilot, eval_m


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json_canonical(path: Path, obj) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return sha256_bytes(raw)
