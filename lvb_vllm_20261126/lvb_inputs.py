"""Strictly label-blind category-balanced 150-sample loader and video reader for LongVideoBench."""

from collections import Counter, defaultdict
from pathlib import Path
from unicodedata import normalize

from missing_fact_routing_20261124.target import ATTRIBUTES, target_from_question


LVB_PARQUET = Path(
    "/data1/llj/development/lljvideo/lljmas/LatentMAS/dataset/LongVideoBench-Meta/data/validation-00000-of-00001.parquet"
)
LVB_VIDEOS = Path("/data1/llj/development/lljvideo/CLAUDEvidoe/LongVideoBench303/videos")
PUBLIC_COLUMNS = (
    "id",
    "video_id",
    "video_path",
    "question",
    "option0",
    "option1",
    "option2",
    "option3",
    "option4",
    "duration",
    "duration_group",
    "question_category",
    "topic_category",
)
N_SAMPLES = 150
MIN_DURATION_S = 180.0


def select_category_balanced_cases(rows, *, n_total=N_SAMPLES, min_duration=MIN_DURATION_S):
    """Select n_total samples balanced across all question categories (max unique & long videos first).

    Strictly deterministic and label-blind (never accesses ``correct_choice``).
    """
    eligible = [r for r in rows if float(r["duration"]) >= min_duration]
    by_cat = defaultdict(list)
    for r in eligible:
        by_cat[r["question_category"]].append(r)

    cats = sorted(by_cat)
    if not cats:
        raise ValueError("No eligible LVB rows found")
    base_q, rem = divmod(n_total, len(cats))
    # Give +1 quota to categories with more unique videos first (tiebreak alphabetically)
    cat_order = sorted(
        cats,
        key=lambda c: (-len({r["video_path"] for r in by_cat[c]}), -len(by_cat[c]), c),
    )
    quotas = {c: base_q + (1 if i < rem else 0) for i, c in enumerate(cat_order)}
    for c, q in quotas.items():
        if len(by_cat[c]) < q:
            raise ValueError(f"Category {c} only has {len(by_cat[c])} rows, need {q}")

    # Deterministic priority within each category: duration>=600 first, then longer duration, then id
    for c in cats:
        by_cat[c].sort(key=lambda r: (0 if float(r["duration"]) >= 600.0 else 1, -float(r["duration"]), str(r["id"])))

    # Pick round-by-round across categories (scarcest unique-video categories first) to maximize unique videos
    picked_by_cat = {c: [] for c in cats}
    used_vids = Counter()
    used_ids = set()
    pick_order = sorted(cats, key=lambda c: (len({r["video_path"] for r in by_cat[c]}), c))

    max_q = max(quotas.values())
    for step in range(max_q):
        for c in pick_order:
            if len(picked_by_cat[c]) >= quotas[c]:
                continue
            # Prefer unused videos first, then least-used videos, preserving duration>=600 priority
            candidates = [r for r in by_cat[c] if r["id"] not in used_ids]
            candidates.sort(
                key=lambda r: (
                    used_vids[r["video_path"]],
                    0 if float(r["duration"]) >= 600.0 else 1,
                    -float(r["duration"]),
                    str(r["id"]),
                )
            )
            chosen = candidates[0]
            picked_by_cat[c].append(chosen)
            used_vids[chosen["video_path"]] += 1
            used_ids.add(chosen["id"])

    # Interleave round-robin across sorted categories so even/odd 2-GPU shards stay identically balanced
    ordered = []
    for step in range(max_q):
        for c in cats:
            if step < len(picked_by_cat[c]):
                ordered.append(picked_by_cat[c][step])

    if len(ordered) != n_total or len({r["id"] for r in ordered}) != n_total:
        raise ValueError("Category-balanced cohort construction failed")
    return ordered


def load_blind_lvb_cases(*, parquet_path=LVB_PARQUET, video_dir=LVB_VIDEOS, n_total=N_SAMPLES):
    """Load n_total category-balanced LVB validation samples without reading correct_choice."""
    import pyarrow.parquet as pq

    video_dir = Path(video_dir)
    mp4s = {p.name: p for p in video_dir.glob("*.mp4")}
    # Physically read ONLY PUBLIC_COLUMNS (correct_choice is never read!)
    tbl = pq.read_table(parquet_path, columns=list(PUBLIC_COLUMNS)).to_pydict()
    n = len(tbl["id"])

    rows = []
    for i in range(n):
        vp = tbl["video_path"][i]
        if vp not in mp4s:
            continue
        choices = [
            str(tbl[f"option{k}"][i]).strip()
            for k in range(5)
            if tbl[f"option{k}"][i] not in (None, "", "N/A") and str(tbl[f"option{k}"][i]).strip()
        ]
        q = str(tbl["question"][i]).strip()
        if not q or len(choices) < 2:
            continue
        full_path = mp4s[vp]
        rows.append(dict(
            id=str(tbl["id"][i]),
            video_uid=str(tbl["video_id"][i]),
            video_path=str(full_path),
            video_name=vp,
            video_bytes=full_path.stat().st_size,
            question=q,
            choices=choices,
            duration=float(tbl["duration"][i]),
            duration_annotated=float(tbl["duration"][i]),
            duration_group=int(tbl["duration_group"][i]),
            question_category=str(tbl["question_category"][i]),
            topic_category=str(tbl["topic_category"][i]),
        ))

    selected = select_category_balanced_cases(rows, n_total=n_total)
    cases = []
    for idx, r in enumerate(selected):
        cases.append(dict(
            r,
            ordinal=idx,
            seed_index=idx,
            split="lvb_eval",
        ))
    return cases


def abi_case(case):
    """Exact acquisition_contract schema view without mutating provenance."""
    return dict(
        id=case["id"],
        video_uid=case["video_uid"],
        question=case["question"],
        choices=case["choices"],
        duration_annotated=case["duration_annotated"],
        video_path=case["video_path"],
        video_bytes=case["video_bytes"],
        split="development",
    )


def plan_lvb_foils(cases, tokenizer):
    """Exact same deterministic label-blind foil matcher as CG-Bench."""
    recognized = {name for name, _ in ATTRIBUTES}
    profiles = {}
    for case in cases:
        cid, uid = case["id"], case["video_uid"]
        target = target_from_question(case["question"])
        fields = (target.question, target.entity, target.attribute, target.time_condition)
        count = len(tokenizer.encode("\n".join(fields), add_special_tokens=False))
        qkey = " ".join(normalize("NFKC", target.question).casefold().split()).rstrip(" ?")
        profiles[cid] = dict(
            video=uid, question=target.question, key=qkey,
            attribute=target.attribute, tokens=count,
        )
    plan = {}
    for cid, actual in sorted(profiles.items()):
        eligible = [
            other for other, foil in profiles.items()
            if foil["video"] != actual["video"] and foil["key"] != actual["key"]
        ]
        if not eligible:
            raise ValueError(f"No distinct foil question for {cid}")

        def rank(other):
            foil = profiles[other]
            mismatch = actual["attribute"] in recognized and actual["attribute"] != foil["attribute"]
            return mismatch, abs(actual["tokens"] - foil["tokens"]), other

        foil_id = min(eligible, key=rank)
        foil = profiles[foil_id]
        plan[cid] = dict(
            foil_id=foil_id, question=foil["question"],
            attribute=actual["attribute"], foil_attribute=foil["attribute"],
            tokens=actual["tokens"], foil_tokens=foil["tokens"],
        )
    return plan


class VideoFrameReader:
    """Fast OpenCV random-access MP4 reader with PyAV fallback."""

    def __init__(self, path):
        import cv2

        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.frame_count <= 0 or self.fps <= 0:
            self.cap.release()
            raise RuntimeError(f"Invalid video geometry ({self.frame_count}, {self.fps}): {self.path}")

    def read_pil_frames(self, indices):
        import cv2
        from PIL import Image

        images = []
        missing = []
        for pos, idx in enumerate(indices):
            clamped = max(0, min(self.frame_count - 1, int(idx)))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, clamped)
            ok, bgr = self.cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                images.append(Image.fromarray(rgb).convert("RGB"))
            else:
                images.append(None)
                missing.append((pos, clamped))

        if missing:
            # PyAV timestamp-seek fallback for rare frames where OpenCV seek misses
            import av

            with av.open(self.path) as container:
                stream = container.streams.video[0]
                for pos, clamped in missing:
                    target_sec = clamped / self.fps
                    pts = int(target_sec / float(stream.time_base))
                    container.seek(max(0, pts), stream=stream, backward=True)
                    img = None
                    for frame in container.decode(stream):
                        img = frame.to_image().convert("RGB")
                        if frame.time is None or frame.time >= target_sec - (0.5 / self.fps):
                            break
                    if img is None:
                        raise RuntimeError(f"Both cv2 and av failed to decode frame {clamped} in {self.path}")
                    images[pos] = img
        return images

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
