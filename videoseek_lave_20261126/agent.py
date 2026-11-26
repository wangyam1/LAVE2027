"""Multi-step VideoSeek + LAVE agent runtime (100% training-free, single-video self-controlled, 0 vision replay)."""

from dataclasses import dataclass, field
import json
import math
import re
from statistics import median
from time import perf_counter

import numpy as np
from PIL import Image

from .prompts import (
    ACTION_PARSER_PROMPT,
    VIDEOSEEK_SYSTEM_PROMPT,
    build_videoseek_tools,
    format_mc_question,
    render_latent_readout_prompt,
    render_lave_receiver_prompt,
)


def decode_frames_pyav(video_path, timestamps_s, *, short_side=None, max_pixels=None):
    """Decode exact frames closest to requested timestamps (seconds) via deadlock-free OpenCV VideoCapture."""
    import cv2

    if not timestamps_s:
        return [], []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (total_frames / fps) if total_frames > 0 else max(timestamps_s) + 1.0

        images = []
        actual_ts = []
        for req_t in timestamps_s:
            t = max(0.0, min(float(req_t), max(0.0, dur - 0.05)))
            frame_idx = int(round(t * fps))
            if total_frames > 0:
                frame_idx = max(0, min(frame_idx, total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                # Fallback: step slightly backward if near EOF
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx - 5))
                ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb)
                fts = frame_idx / fps
            else:
                img = Image.new("RGB", (320, 240), (0, 0, 0))
                fts = t
            img = _resize_image(img, short_side=short_side, max_pixels=max_pixels)
            images.append(img)
            actual_ts.append(round(fts, 2))
        return images, actual_ts
    finally:
        cap.release()


def _resize_image(img: Image.Image, *, short_side=None, max_pixels=None) -> Image.Image:
    w, h = img.size
    if short_side is not None and min(w, h) > 0:
        scale = float(short_side) / float(min(w, h))
        nw = max(28, int(round(w * scale)))
        nh = max(28, int(round(h * scale)))
        img = img.resize((nw, nh), Image.BICUBIC)
        w, h = img.size
    if max_pixels is not None and w * h > max_pixels:
        scale = math.sqrt(float(max_pixels) / float(w * h))
        nw = max(28, int(w * scale))
        nh = max(28, int(h * scale))
        img = img.resize((nw, nh), Image.BICUBIC)
    return img


def uniform_timestamps(start_s, end_s, num_frames, duration):
    s = max(0.0, min(float(start_s), max(0.0, float(duration) - 0.1)))
    e = max(s + 0.1, min(float(end_s), float(duration)))
    if num_frames <= 1:
        return [round(0.5 * (s + e), 2)]
    return [round(float(x), 2) for x in np.linspace(s, e, int(num_frames))]


def strip_think_tags(text):
    """Strip <think>...</think> reasoning blocks emitted by Thinking models cleanly."""
    s = (text or "").strip()
    if "</think>" in s:
        after = s.rsplit("</think>", 1)[1].strip()
        if after:
            return after
        # If nothing after </think>, return content inside <think> stripped of tags
        return re.sub(r"</?think>", "", s).strip()
    if "<think>" in s:
        return re.sub(r"</?think>", "", s).strip()
    return s


def choose_geometric_clock_peak(duration, planner_start, planner_end, visited_windows=()):
    """Deterministic equal-budget geometric peak: midpoint of the largest unvisited temporal gap outside planner window."""
    dur = max(2.0, float(duration))
    intervals = [(max(0.0, float(planner_start)), min(dur, float(planner_end)))]
    for lo, hi in visited_windows:
        intervals.append((max(0.0, float(lo)), min(dur, float(hi))))
    intervals.sort()
    merged = []
    for lo, hi in intervals:
        if not merged or lo > merged[-1][1]:
            merged.append([lo, hi])
        else:
            merged[-1][1] = max(merged[-1][1], hi)

    gaps = []
    cur = 0.0
    for lo, hi in merged:
        if lo > cur + 1.0:
            gaps.append((lo - cur, 0.5 * (cur + lo)))
        cur = max(cur, hi)
    if dur > cur + 1.0:
        gaps.append((dur - cur, 0.5 * (cur + dur)))

    if gaps:
        gaps.sort(key=lambda x: x[0], reverse=True)
        return [round(float(gaps[0][1]), 2)]
    # Fallback: opposite side of video outside planner window
    mid_p = 0.5 * (float(planner_start) + float(planner_end))
    cand = 0.25 * dur if mid_p >= 0.5 * dur else 0.75 * dur
    if float(planner_start) <= cand <= float(planner_end):
        return []
    return [round(float(cand), 2)]


def parse_tool_call(raw_text, valid_tools=("overview", "skim", "focus", "answer")):
    """Parse Qwen3.5 <tool_call> XML or JSON tool call deterministically."""
    text = strip_think_tags(raw_text)
    # Pattern 1: Qwen3.5 XML format <tool_call><function=name><parameter=k>v</parameter>...</function></tool_call>
    m_fn = re.search(r"<function\s*=\s*([a-zA-Z0-9_]+)\s*>([\s\S]*?)</function>", text)
    if m_fn:
        name = m_fn.group(1).strip().lower()
        body = m_fn.group(2)
        params = {}
        for pm in re.finditer(r"<parameter\s*=\s*([a-zA-Z0-9_]+)\s*>([\s\S]*?)</parameter>", body):
            k = pm.group(1).strip()
            v = pm.group(2).strip()
            if k in ("start_time", "end_time"):
                try:
                    params[k] = float(re.sub(r"[^0-9.\-]", "", v))
                except ValueError:
                    pass
            else:
                params[k] = v
        if name in valid_tools:
            return dict(name=name, arguments=params, parse_mode="qwen_xml")

    # Pattern 2: <tool_call>{"name": ..., "arguments": ...}</tool_call>
    for m_json in re.finditer(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", text):
        try:
            obj = json.loads(m_json.group(1))
            name = str(obj.get("name", "")).strip().lower()
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name in valid_tools and isinstance(args, dict):
                return dict(name=name, arguments=args, parse_mode="tool_call_json")
        except Exception:
            pass

    # Pattern 3: bare JSON object with "name"
    m_bare = re.search(r"\{\s*\"name\"\s*:\s*\"([a-zA-Z0-9_]+)\"[\s\S]*?\}", text)
    if m_bare:
        try:
            obj = json.loads(m_bare.group(0))
            name = str(obj.get("name", "")).strip().lower()
            args = obj.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if name in valid_tools and isinstance(args, dict):
                return dict(name=name, arguments=args, parse_mode="bare_json")
        except Exception:
            pass

    # Pattern 4: function-like text fallback e.g. focus(query="...", start_time=10, end_time=25)
    for tname in ("focus", "skim", "overview", "answer"):
        if re.search(rf"\b{tname}\s*\(", text, flags=re.IGNORECASE):
            if tname in ("overview", "answer"):
                return dict(name=tname, arguments={}, parse_mode="regex_call")
            st = re.search(r"start_time\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text)
            et = re.search(r"end_time\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text)
            qm = re.search(r"query\s*[=:]\s*[\"']([^\"']+)[\"']", text)
            if st and et:
                return dict(
                    name=tname,
                    arguments=dict(
                        query=qm.group(1) if qm else "Find evidence relevant to the question.",
                        start_time=float(st.group(1)),
                        end_time=float(et.group(1)),
                    ),
                    parse_mode="regex_call",
                )
    return None


def extract_choice_letter(text, valid_letters):
    """Extract a single valid uppercase multiple-choice letter from model output."""
    s = strip_think_tags(text)
    valid_set = set(valid_letters)
    if len(s) == 1 and s.upper() in valid_set:
        return s.upper()
    # Look for explicit answer patterns first
    for pat in (
        r"(?:final\s+answer|answer|option)\s*(?:is|:)?\s*[\(\[\*]*([A-Z])[\)\]\*\.]?\b",
        r"^\s*[\(\[]?([A-Z])[\)\]\.:]\s*",
        r"\b([A-Z])\s*$",
    ):
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m and m.group(1).upper() in valid_set:
            return m.group(1).upper()
    # Fallback: first standalone valid letter
    for ch in re.findall(r"\b([A-Z])\b", s.upper()):
        if ch in valid_set:
            return ch
    return None


def normalized_positive_entropy(lifts):
    """Normalized Shannon entropy over positive lifts (0 = concentrated peak, 1 = dispersed)."""
    pos = [max(0.0, float(v)) for v in lifts]
    tot = sum(pos)
    if tot <= 1e-9:
        return 1.0
    probs = [v / tot for v in pos if v > 0.0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(pos))


def select_latent_peaks(timestamps_s, lifts, *, min_sep_s=12.0, max_peaks=2, excluded_intervals=()):
    """Select non-overlapping highest-lift timestamps via NMS, preferring unvisited regions."""
    order = sorted(range(len(lifts)), key=lambda i: float(lifts[i]), reverse=True)
    chosen = []
    for idx in order:
        if float(lifts[idx]) <= 0.0:
            break
        t = float(timestamps_s[idx])
        if any(lo <= t <= hi for lo, hi in excluded_intervals):
            continue
        if any(abs(t - float(timestamps_s[c])) < min_sep_s for c in chosen):
            continue
        chosen.append(idx)
        if len(chosen) >= max_peaks:
            break
    return chosen


def plan_split_budget_timestamps(
    tool_name,
    planner_start,
    planner_end,
    duration,
    total_frames,
    peak_timestamps,
    *,
    focus_half_window=5.0,
    skim_half_window=14.0,
):
    """Strictly preserve total_frames while splitting half for Planner anchor and half for LAVE latent peak(s).

    If all peak_timestamps already fall inside [planner_start, planner_end], returns planner timestamps unchanged.
    """
    p_start = max(0.0, min(float(planner_start), max(0.0, duration - 1.0)))
    p_end = max(p_start + 1.0, min(float(planner_end), float(duration)))

    outside_peaks = [t for t in peak_timestamps if not (p_start <= t <= p_end)]
    if not outside_peaks:
        ts = uniform_timestamps(p_start, p_end, total_frames, duration)
        return dict(
            rerouted=False,
            timestamps=ts,
            planner_window=(round(p_start, 1), round(p_end, 1)),
            latent_windows=[],
        )

    n_planner = total_frames // 2
    n_latent = total_frames - n_planner
    half_w = focus_half_window if tool_name == "focus" else skim_half_window

    planner_ts = uniform_timestamps(p_start, p_end, n_planner, duration)
    latent_ts = []
    latent_windows = []
    k = min(len(outside_peaks), 2)
    per_peak = [n_latent // k + (1 if i < (n_latent % k) else 0) for i in range(k)]
    for t_peak, nf in zip(outside_peaks[:k], per_peak):
        lo = max(0.0, t_peak - half_w)
        hi = min(float(duration), t_peak + half_w)
        if hi - lo < 1.0:
            hi = min(float(duration), lo + 2.0)
        latent_windows.append((round(lo, 1), round(hi, 1)))
        latent_ts.extend(uniform_timestamps(lo, hi, nf, duration))

    merged = sorted(planner_ts + latent_ts)
    if len(merged) != total_frames:
        raise RuntimeError(f"Split budget mismatch: expected {total_frames}, got {len(merged)}")
    return dict(
        rerouted=True,
        timestamps=merged,
        planner_window=(round(p_start, 1), round(p_end, 1)),
        latent_windows=latent_windows,
    )


def select_top_bank_entries(bank, lifts, *, max_frames=16, dedup_tol_s=1.5):
    """Select top-M highest-lift cached visual-state entries sorted chronologically.

    When two entries fall within dedup_tol_s, prefer higher-resolution `focus` > `skim` > `overview`.
    """
    if not bank:
        return []
    tool_rank = {"focus": 3, "skim": 2, "overview": 1}
    scored = []
    for idx, (entry, lift) in enumerate(zip(bank, lifts)):
        scored.append((float(lift), tool_rank.get(entry["tool"], 0), idx))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)

    picked = []
    for lift, tr, idx in scored:
        cand = bank[idx]
        ts = float(cand["timestamp_s"])
        too_close = False
        for p_i, p_idx in enumerate(picked):
            p_ent = bank[p_idx]
            if abs(ts - float(p_ent["timestamp_s"])) <= dedup_tol_s:
                too_close = True
                if tr > tool_rank.get(p_ent["tool"], 0) and lift >= float(lifts[p_idx]) - 0.5:
                    picked[p_i] = idx
                break
        if not too_close and len(picked) < max_frames:
            picked.append(idx)

    picked.sort(key=lambda i: float(bank[i]["timestamp_s"]))
    return picked


@dataclass
class AgentConfig:
    max_steps: int = 6
    overview_frames: int = 32
    skim_frames: int = 16
    focus_frames: int = 16
    focus_max_span: float = 24.0
    overview_short_side: int = 256
    skim_short_side: int = 256
    focus_max_pixels: int = 262144
    latent_qa_frames: int = 8
    max_bank_scan: int = 48
    specificity_margin: float = 0.25
    overview_max_tokens: int = 768
    skim_max_tokens: int = 640
    focus_max_tokens: int = 512
    thought_max_tokens: int = 384
    thinking_max_tokens: int = 1024
    enable_thinking: bool = False
    parser_max_tokens: int = 160
    answer_max_tokens: int = 32


_GLOBAL_VLM_CALL_SEQ = 0


def _next_vlm_call_uid():
    global _GLOBAL_VLM_CALL_SEQ
    _GLOBAL_VLM_CALL_SEQ += 1
    return _GLOBAL_VLM_CALL_SEQ


class VideoSeekLaveRunner:
    """Runs matched Vanilla VideoSeek and VideoSeek + LAVE trajectories on a single video."""

    def __init__(self, engine, config: AgentConfig | None = None):
        self.engine = engine
        self.cfg = config or AgentConfig()
        self.tools = build_videoseek_tools(
            overview_frames=self.cfg.overview_frames,
            skim_frames=self.cfg.skim_frames,
            focus_frames=self.cfg.focus_frames,
            focus_max_span=int(self.cfg.focus_max_span),
        )
        self.system_prompt = VIDEOSEEK_SYSTEM_PROMPT.format(
            overview_num_frames=self.cfg.overview_frames,
            skim_num_frames=self.cfg.skim_frames,
            focus_num_frames=self.cfg.focus_frames,
            focus_max_span=int(self.cfg.focus_max_span),
        )
        self._vlm_call_seq = 0

    def run_shared_overview(self, video_path, duration):
        """Execute official Step-1 overview once and capture 32 post-merger visual slots for free."""
        ts_req = uniform_timestamps(0.0, duration, self.cfg.overview_frames, duration)
        images, ts_actual = decode_frames_pyav(
            video_path, ts_req, short_side=self.cfg.overview_short_side, max_pixels=self.cfg.focus_max_pixels
        )
        ov_uid = _next_vlm_call_uid()
        self._vlm_call_seq = ov_uid
        content = [
            {
                "type": "text",
                "text": (
                    f"[Overview #{ov_uid}] The video spans 0.0s - {duration:.1f}s. Below are {len(images)} uniformly sampled frames "
                    "with their exact timestamps.\n"
                ),
            }
        ]
        for img, ts in zip(images, ts_actual):
            content.append({"type": "text", "text": f"{ts:.1f}s:"})
            content.append({"type": "image"})
        content.append(
            {
                "type": "text",
                "text": (
                    "Provide a concise chronological timeline summarizing key scenes, people, objects, actions, "
                    "and visible on-screen text with their exact timestamps (e.g., `12.3s: ...`). "
                    "Cover the full video so an agent knows which timestamps to inspect next."
                ),
            }
        )
        messages = [{"role": "user", "content": content}]
        out = self.engine.generate_vlm_with_capture(
            messages, images, max_tokens=self.cfg.overview_max_tokens
        )
        bank_entries = [
            dict(
                slot=out["slots"][i],
                grid=out["grids"][i],
                timestamp_s=ts_actual[i],
                tool="overview",
                step_id=1,
            )
            for i in range(len(ts_actual))
        ]
        return dict(
            text=out["text"].strip(),
            timestamps=ts_actual,
            bank_entries=bank_entries,
            frames=len(ts_actual),
            vision_calls=out["vision_calls"],
            seconds=out["seconds"],
        )

    def _exec_skim_or_focus(
        self,
        tool_name,
        video_path,
        duration,
        query,
        timestamps_s,
        step_id,
        window_note,
        *,
        tool_cache=None,
    ):
        cache_key = (tool_name, tuple(round(float(t), 2) for t in timestamps_s), query.strip())
        if tool_cache is not None and cache_key in tool_cache:
            cached = tool_cache[cache_key]
            reused_entries = [dict(e, step_id=step_id) for e in cached["bank_entries"]]
            return dict(
                text=cached["text"],
                timestamps=list(cached["timestamps"]),
                bank_entries=reused_entries,
                frames=cached["frames"],
                vision_calls=0,
                seconds=0.0,
                cache_reused=True,
            )

        if tool_name == "skim":
            images, ts_actual = decode_frames_pyav(
                video_path,
                timestamps_s,
                short_side=self.cfg.skim_short_side,
                max_pixels=self.cfg.focus_max_pixels,
            )
            prompt_tail = (
                f"Question:\n{query}\n\n"
                "Describe the relevant scenes concisely with exact timestamps. "
                "Highlight which timestamps contain visual evidence related to the question and explain why. "
                "If nothing matches, reply 'No relevant content found.'"
            )
            max_tok = self.cfg.skim_max_tokens
        else:
            images, ts_actual = decode_frames_pyav(
                video_path,
                timestamps_s,
                short_side=None,
                max_pixels=self.cfg.focus_max_pixels,
            )
            prompt_tail = (
                f"Question:\n{query}\n\n"
                "Inspect the fine visual details (people, objects, actions, text, expressions) carefully and "
                "answer the query citing exact timestamps. "
                "If the clip is not related to the question, return 'No relevant content found.'"
            )
            max_tok = self.cfg.focus_max_tokens

        call_uid = _next_vlm_call_uid()
        self._vlm_call_seq = call_uid
        content = [{"type": "text", "text": f"[Clip #{call_uid}] Inspected video frames ({window_note}):\n"}]
        for img, ts in zip(images, ts_actual):
            content.append({"type": "text", "text": f"{ts:.1f}s:"})
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt_tail})

        out = self.engine.generate_vlm_with_capture(
            [{"role": "user", "content": content}], images, max_tokens=max_tok
        )
        entries = [
            dict(
                slot=out["slots"][i],
                grid=out["grids"][i],
                timestamp_s=ts_actual[i],
                tool=tool_name,
                step_id=step_id,
            )
            for i in range(len(ts_actual))
        ]
        res = dict(
            text=out["text"].strip(),
            timestamps=ts_actual,
            bank_entries=entries,
            frames=len(ts_actual),
            vision_calls=out["vision_calls"],
            seconds=out["seconds"],
            cache_reused=False,
        )
        if tool_cache is not None:
            tool_cache[cache_key] = res
        return res

    def _score_bank_and_controls(self, bank, question_text, query, trajectory_brief, seed_idx):
        """Run zero-pixel Calibrated-Lift Receiver + pure single-video self-controls on bank."""
        if not bank:
            return None
        # Subsample uniformly if bank exceeds max_bank_scan, always keeping all recent focus frames
        if len(bank) > self.cfg.max_bank_scan:
            focus_idx = [i for i, e in enumerate(bank) if e["tool"] == "focus"]
            other_idx = [i for i, e in enumerate(bank) if e["tool"] != "focus"]
            keep_other = max(0, self.cfg.max_bank_scan - len(focus_idx))
            if keep_other > 0 and other_idx:
                pick_o = [other_idx[int(x)] for x in np.linspace(0, len(other_idx) - 1, min(keep_other, len(other_idx)))]
            else:
                pick_o = []
            scan_indices = sorted(set(focus_idx[-self.cfg.max_bank_scan:] + pick_o))
        else:
            scan_indices = list(range(len(bank)))

        scan_slots = [bank[i]["slot"] for i in scan_indices]
        scan_grids = [bank[i]["grid"] for i in scan_indices]
        scan_ts = [bank[i]["timestamp_s"] for i in scan_indices]

        prompt_text, (y_id, n_id) = render_lave_receiver_prompt(
            self.engine.tokenizer, question_text, query, trajectory_brief
        )
        res = self.engine.score_mixed_grid_slots(prompt_text, scan_slots, scan_grids, y_id, n_id)
        if res["visual_forwards"] != 0:
            raise RuntimeError("Receiver violated zero-vision-forward invariant!")

        lifts = res["lifts"]
        best_local = max(range(len(lifts)), key=lambda i: lifts[i])
        best_lift = float(lifts[best_local])
        med_lift = float(median(lifts))
        ent = normalized_positive_entropy(lifts)

        # Single-video Control 1: lift > 0
        c_pos = best_lift > 0.0
        # Single-video Control 3: temporal specificity over video median
        c_spec = best_lift > (med_lift + self.cfg.specificity_margin)

        # Single-video Control 2: semantic content vs scrambled same-norm noise (only run if c_pos)
        scram_lift = None
        scram_s = 0.0
        scram_fwds = 0
        c_scram = False
        if c_pos:
            sc = self.engine.score_scrambled_control(
                prompt_text, scan_slots[best_local], scan_grids[best_local], y_id, n_id, seed=16423 + seed_idx
            )
            if sc["visual_forwards"] != 0:
                raise RuntimeError("Scrambled control violated zero-vision-forward invariant!")
            gkey = tuple(int(v) for v in scan_grids[best_local])
            zy0, zn0 = res["zero_by_grid"][gkey]
            scram_lift = (sc["score"][0] - sc["score"][1]) - (zy0 - zn0)
            c_scram = best_lift > scram_lift
            scram_s = sc["seconds"]
            scram_fwds = sc["forwards"]

        approved = bool(c_pos and c_scram and c_spec)
        return dict(
            scan_indices=scan_indices,
            scan_timestamps=scan_ts,
            lifts=[round(float(x), 4) for x in lifts],
            best_local=best_local,
            best_timestamp=scan_ts[best_local],
            best_lift=round(best_lift, 4),
            median_lift=round(med_lift, 4),
            scrambled_lift=round(float(scram_lift), 4) if scram_lift is not None else None,
            entropy=round(float(ent), 4),
            controls=dict(positive=c_pos, scrambled=c_scram, specific=c_spec),
            approved=approved,
            forwards=res["forwards"] + scram_fwds,
            visual_forwards=0,
            seconds=round(res["seconds"] + scram_s, 3),
        )

    def _run_latent_readout_answer(
        self, bank, question_text, valid_letters, trajectory_brief, *, planner_reasoning=""
    ):
        """Mechanism B: zero-pixel visual verification directly over top-M cached post-merger states."""
        # Score entire bank against the question + options using 0 vision calls
        prompt_rx, (y_id, n_id) = render_lave_receiver_prompt(
            self.engine.tokenizer,
            question_text,
            "Select the frames showing direct visual evidence needed to choose the right answer option.",
            trajectory_brief,
        )
        slots_all = [e["slot"] for e in bank]
        grids_all = [e["grid"] for e in bank]
        score_res = self.engine.score_mixed_grid_slots(prompt_rx, slots_all, grids_all, y_id, n_id)
        if score_res["visual_forwards"] != 0:
            raise RuntimeError("Latent readout selection violated zero-vision invariant!")

        picked_idx = select_top_bank_entries(
            bank, score_res["lifts"], max_frames=self.cfg.latent_qa_frames, dedup_tol_s=1.5
        )
        sel_slots = [bank[i]["slot"] for i in picked_idx]
        sel_grids = [bank[i]["grid"] for i in picked_idx]
        sel_meta = [dict(timestamp_s=bank[i]["timestamp_s"], tool=bank[i]["tool"]) for i in picked_idx]

        qa_prompt = render_latent_readout_prompt(
            self.engine.tokenizer,
            question_text,
            valid_letters,
            sel_meta,
            trajectory_brief,
            planner_reasoning=planner_reasoning,
        )
        gen = self.engine.generate_from_visual_embeds(
            qa_prompt, sel_slots, sel_grids, max_tokens=self.cfg.answer_max_tokens
        )
        if gen["visual_forwards"] != 0:
            raise RuntimeError("Latent readout answering violated zero-vision invariant!")
        choice = extract_choice_letter(gen["text"], valid_letters)
        return dict(
            answer=choice,
            raw_text=gen["text"].strip(),
            selected_timestamps=[m["timestamp_s"] for m in sel_meta],
            selected_tools=[m["tool"] for m in sel_meta],
            num_frames=gen["num_frames"],
            visual_tokens=gen["visual_tokens"],
            receiver_forwards=score_res["forwards"],
            visual_forwards=0,
            seconds=round(score_res["seconds"] + gen["seconds"], 3),
        )

    def _run_trajectory(self, case, shared_overview, *, use_lave_routing: bool, seed_base: int, tool_cache=None):
        """Run one full multi-step VideoSeek trajectory (vanilla or LAVE-routed) + latent readout."""
        t_start = perf_counter()
        question_text, valid_letters = format_mc_question(case["question"], case["options"])
        duration = float(case["duration"])
        video_path = case["video_path"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Video Duration: {duration:.1f}s\n\n"
                    f"Question:\n{question_text}"
                ),
            },
        ]
        bank = list(shared_overview["bank_entries"])
        steps_log = []
        visited_focus_windows = []
        seen_calls = set()
        total_frames = shared_overview["frames"]
        total_vision_calls = shared_overview["vision_calls"]
        visual_tool_calls = 1  # shared overview
        receiver_seconds = 0.0
        receiver_forwards = 0
        reroute_count = 0
        control_scans = 0
        control_approvals = 0

        # Step 1 is the shared overview (official VideoSeek always opens with overview when no subtitles exist)
        ov_thought = (
            "No prior visual observations are available yet. "
            "I will call `overview` to get a structured timeline of the entire video."
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Step [1 / {self.cfg.max_steps}]: Follow the Thinking Policy and plan the next action."
                ),
            }
        )
        messages.append({"role": "assistant", "content": ov_thought})
        messages.append(
            {
                "role": "user",
                "content": (
                    f"Observation from `overview()` (0.0s - {duration:.1f}s, {shared_overview['frames']} frames):\n"
                    f"{shared_overview['text']}"
                ),
            }
        )
        steps_log.append(
            dict(
                step_id=1,
                tool="overview",
                arguments={},
                frames=shared_overview["frames"],
                rerouted=False,
                seconds=shared_overview["seconds"],
            )
        )

        final_text_answer = None
        final_text_raw = ""
        step4_text_answer = None
        step4_frames = None
        finish_reason = "reach_max_steps"

        for step_id in range(2, self.cfg.max_steps + 1):
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Step [{step_id} / {self.cfg.max_steps}]: "
                        "Briefly summarize what evidence you have, identify any missing visual detail, "
                        "and state ONE next tool call (`skim`, `focus`, or `answer`) with exact start_time/end_time in seconds."
                    ),
                }
            )
            t_max = self.cfg.thinking_max_tokens if self.cfg.enable_thinking else self.cfg.thought_max_tokens
            if self.cfg.enable_thinking:
                thought_out = self.engine.generate_text(messages, max_tokens=t_max, enable_thinking=True)
            else:
                thought_out = self.engine.generate_text(messages, max_tokens=t_max)
            thought = strip_think_tags(thought_out["text"]) or "Need more visual evidence."
            messages.append({"role": "assistant", "content": thought})

            # Action parser (official VideoSeek 2-stage thought -> tool_call)
            recent_obs = "\n\n".join(
                m["content"][:1000] for m in messages if m["role"] == "user" and m["content"].startswith("Observation")
            )[-2000:]
            parse_msgs = [
                {"role": "system", "content": ACTION_PARSER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Video Duration: {duration:.1f}s\n\n"
                        f"Question:\n{question_text}\n\n"
                        f"Planner Thought:\n{thought}\n\n"
                        f"Latest Observations:\n{recent_obs}\n\n"
                        "Return exactly ONE function call."
                    ),
                },
            ]
            parse_out = self.engine.generate_text(
                parse_msgs, max_tokens=self.cfg.parser_max_tokens, tools=self.tools
            )
            action = parse_tool_call(parse_out["text"])
            if action is None:
                action = parse_tool_call(thought)
            if action is None:
                # Fallback: if thought clearly names a final letter or we're at last step -> answer
                if step_id == self.cfg.max_steps:
                    action = dict(name="answer", arguments={}, parse_mode="max_step_fallback")
                else:
                    steps_log.append(
                        dict(step_id=step_id, tool="invalid", raw=parse_out["text"][:200], frames=0, rerouted=False)
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": "Invalid tool call. You MUST invoke exactly one tool: skim, focus, or answer.",
                        }
                    )
                    continue

            tname = action["name"]
            args = dict(action.get("arguments") or {})

            if tname == "answer":
                ans_msgs = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            f"Question:\n{question_text}\n\n"
                            f"Respond with ONLY the single best option letter ({', '.join(valid_letters)})."
                        ),
                    }
                ]
                ans_out = self.engine.generate_text(ans_msgs, max_tokens=self.cfg.answer_max_tokens)
                final_text_raw = ans_out["text"].strip()
                final_text_answer = extract_choice_letter(final_text_raw, valid_letters)
                finish_reason = "answer_tool"
                steps_log.append(
                    dict(step_id=step_id, tool="answer", arguments={}, frames=0, rerouted=False, seconds=ans_out["seconds"])
                )
                break

            if tname == "overview":
                # Already ran overview at step 1; nudge planner to narrow down instead of wasting a step
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Observation: `overview` was already executed at Step 1. "
                            "Please use `skim` or `focus` on a specific time range, or call `answer`."
                        ),
                    }
                )
                steps_log.append(dict(step_id=step_id, tool="overview_skip", arguments={}, frames=0, rerouted=False))
                continue

            # Normalize skim / focus arguments
            raw_s = float(args.get("start_time", 0.0))
            raw_e = float(args.get("end_time", min(duration, raw_s + 16.0)))
            if raw_e < raw_s:
                raw_s, raw_e = raw_e, raw_s
            p_start = max(0.0, min(raw_s, max(0.0, duration - 1.0)))
            p_end = max(p_start + 1.0, min(raw_e, duration))
            if tname == "focus" and (p_end - p_start) > self.cfg.focus_max_span:
                mid = 0.5 * (p_start + p_end)
                p_start = max(0.0, mid - 0.5 * self.cfg.focus_max_span)
                p_end = min(duration, p_start + self.cfg.focus_max_span)
            query = str(args.get("query") or case["question"]).strip()

            call_sig = (tname, round(p_start, 0), round(p_end, 0))
            if call_sig in seen_calls and not use_lave_routing:
                # Slight deterministic shift if vanilla repeats exact same interval
                shift = 12.0 if p_end + 12.0 <= duration else -12.0
                p_start = max(0.0, min(duration - 2.0, p_start + shift))
                p_end = max(p_start + 2.0, min(duration, p_end + shift))
            seen_calls.add(call_sig)

            n_frames = self.cfg.focus_frames if tname == "focus" else self.cfg.skim_frames
            lave_audit = None
            rerouted = False

            if use_lave_routing:
                control_scans += 1
                traj_brief = "\n".join(
                    m["content"][:900]
                    for m in messages
                    if m["role"] == "user" and m["content"].startswith("Observation")
                )
                lave_audit = self._score_bank_and_controls(
                    bank, question_text, query, traj_brief, seed_idx=seed_base * 10 + step_id
                )
                receiver_seconds += lave_audit["seconds"]
                receiver_forwards += lave_audit["forwards"]
                if lave_audit["approved"]:
                    control_approvals += 1
                    # Entropy-guided number of latent peaks (1 when concentrated, 2 when spread)
                    max_p = 1 if lave_audit["entropy"] < 0.72 else 2
                    peak_local_ids = select_latent_peaks(
                        lave_audit["scan_timestamps"],
                        lave_audit["lifts"],
                        min_sep_s=12.0,
                        max_peaks=max_p,
                        excluded_intervals=visited_focus_windows,
                    )
                    peak_ts = [lave_audit["scan_timestamps"][i] for i in peak_local_ids]
                    plan = plan_split_budget_timestamps(
                        tname,
                        p_start,
                        p_end,
                        duration,
                        n_frames,
                        peak_ts,
                    )
                    ts_req = plan["timestamps"]
                    rerouted = plan["rerouted"]
                    if rerouted:
                        reroute_count += 1
                        lw_str = ", ".join(f"{lo:.1f}s-{hi:.1f}s" for lo, hi in plan["latent_windows"])
                        window_note = f"Planner {p_start:.1f}s-{p_end:.1f}s + LAVE latent-guided {lw_str}"
                    else:
                        window_note = f"{p_start:.1f}s-{p_end:.1f}s"
                    lave_audit["peak_timestamps"] = peak_ts
                    lave_audit["rerouted"] = rerouted
                    lave_audit["latent_windows"] = plan["latent_windows"]
                else:
                    ts_req = uniform_timestamps(p_start, p_end, n_frames, duration)
                    window_note = f"{p_start:.1f}s-{p_end:.1f}s"
                    lave_audit["rerouted"] = False
            else:
                ts_req = uniform_timestamps(p_start, p_end, n_frames, duration)
                window_note = f"{p_start:.1f}s-{p_end:.1f}s"

            tool_out = self._exec_skim_or_focus(
                tname, video_path, duration, query, ts_req, step_id, window_note, tool_cache=tool_cache
            )
            bank.extend(tool_out["bank_entries"])
            total_frames += tool_out["frames"]
            total_vision_calls += tool_out["vision_calls"]
            visual_tool_calls += 1
            if tname == "focus":
                visited_focus_windows.append((p_start, p_end))
                if lave_audit and lave_audit.get("latent_windows"):
                    visited_focus_windows.extend(lave_audit["latent_windows"])

            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Observation from `{tname}` ({window_note}, query={query!r}):\n"
                        f"{tool_out['text']}"
                    ),
                }
            )
            steps_log.append(
                dict(
                    step_id=step_id,
                    tool=tname,
                    arguments=dict(query=query, start_time=round(p_start, 1), end_time=round(p_end, 1)),
                    window_note=window_note,
                    frames=tool_out["frames"],
                    rerouted=rerouted,
                    lave_audit=lave_audit,
                    obs_preview=tool_out["text"][:300],
                    seconds=tool_out["seconds"],
                )
            )
            # Anytime Step-4 checkpoint (official VideoSeek default max_steps=4) without disturbing trajectory
            if step_id == 4 and step4_text_answer is None and self.cfg.max_steps > 4:
                s4_msgs = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "You have reached the maximum number of steps.\n"
                            f"Question:\n{question_text}\n\n"
                            f"Respond with ONLY the single best option letter ({', '.join(valid_letters)})."
                        ),
                    }
                ]
                s4_out = self.engine.generate_text(s4_msgs, max_tokens=self.cfg.answer_max_tokens)
                step4_text_answer = extract_choice_letter(s4_out["text"].strip(), valid_letters)
                step4_frames = total_frames

        # If max_steps reached without calling answer -> force final pure-text answer (official VideoSeek)
        if final_text_answer is None:
            ans_msgs = list(messages) + [
                {
                    "role": "user",
                    "content": (
                        "You have reached the maximum number of steps.\n"
                        f"Question:\n{question_text}\n\n"
                        f"Respond with ONLY the single best option letter ({', '.join(valid_letters)})."
                    ),
                }
            ]
            ans_out = self.engine.generate_text(ans_msgs, max_tokens=self.cfg.answer_max_tokens)
            final_text_raw = ans_out["text"].strip()
            final_text_answer = extract_choice_letter(final_text_raw, valid_letters)

        if step4_text_answer is None:
            step4_text_answer = final_text_answer
            step4_frames = total_frames

        # Mechanism B: Zero-pixel Latent-Retrieved Visual Verification preserving Planner's multi-step reasoning
        traj_brief = "\n\n".join(
            m["content"][:1200] for m in messages if m["role"] == "user" and m["content"].startswith("Observation")
        )
        planner_thoughts = "\n\n".join(
            m["content"][:600] for m in messages if m["role"] == "assistant"
        )
        latent_qa = self._run_latent_readout_answer(
            bank, question_text, valid_letters, traj_brief, planner_reasoning=planner_thoughts
        )

        return dict(
            use_lave_routing=use_lave_routing,
            finish_reason=finish_reason,
            num_steps=len(steps_log),
            visual_tool_calls=visual_tool_calls,
            total_frames=total_frames,
            step4_frames=step4_frames,
            bank_size=len(bank),
            total_vision_calls=total_vision_calls,
            control_scans=control_scans,
            control_approvals=control_approvals,
            reroute_count=reroute_count,
            receiver_forwards=receiver_forwards,
            receiver_seconds=round(receiver_seconds, 3),
            text_answer=final_text_answer,
            step4_text_answer=step4_text_answer,
            text_answer_raw=final_text_raw,
            latent_qa=latent_qa,
            steps=steps_log,
            elapsed_seconds=round(perf_counter() - t_start, 3),
        )

    def run_case_paired(self, case, ordinal):
        """Run matched Vanilla VideoSeek and VideoSeek + LAVE on one case with shared Step-1 overview."""
        t0 = perf_counter()
        shared_ov = self.run_shared_overview(case["video_path"], float(case["duration"]))
        tool_cache = {}
        vanilla = self._run_trajectory(
            case, shared_ov, use_lave_routing=False, seed_base=ordinal, tool_cache=tool_cache
        )
        lave = self._run_trajectory(
            case, shared_ov, use_lave_routing=True, seed_base=ordinal, tool_cache=tool_cache
        )
        return dict(
            cid=case["id"],
            ordinal=ordinal,
            video_id=case["video_id"],
            question_category=case["question_category"],
            duration=case["duration"],
            shared_overview=dict(
                frames=shared_ov["frames"],
                vision_calls=shared_ov["vision_calls"],
                seconds=shared_ov["seconds"],
                text_preview=shared_ov["text"][:400],
            ),
            vanilla=vanilla,
            lave=lave,
            answers=dict(
                vanilla_text=vanilla["text_answer"],
                vanilla_step4=vanilla["step4_text_answer"],
                vanilla_latent_qa=vanilla["latent_qa"]["answer"],
                lave_route_text=lave["text_answer"],
                lave_step4=lave["step4_text_answer"],
                lave_full=lave["latent_qa"]["answer"],
            ),
            elapsed_seconds=round(perf_counter() - t0, 3),
        )
