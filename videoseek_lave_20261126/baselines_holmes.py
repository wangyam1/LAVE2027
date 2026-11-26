"""Reviewer-preempting baselines on Video-Holmes:
1. Clock-Split (equal-budget 50/50 geometric routing without LAVE visual-state receiver, plus uniform-frame zero-pixel QA).
2. VideoSeek-Image (direct raw-pixel interleaved multi-turn visual agent, no observer captions).
3. Unshared standalone end-to-end latency audit (zero cross-arm overview/tool_cache sharing).
"""

from time import perf_counter
import numpy as np

from .agent import (
    AgentConfig,
    VideoSeekLaveRunner,
    choose_geometric_clock_peak,
    decode_frames_pyav,
    extract_choice_letter,
    parse_tool_call,
    plan_split_budget_timestamps,
    strip_think_tags,
    uniform_timestamps,
)
from .prompts import (
    ACTION_PARSER_PROMPT,
    build_videoseek_tools,
    format_mc_question,
    render_latent_readout_prompt,
)

VIDEOSEEK_IMAGE_SYSTEM_PROMPT = """# Role
You are an efficient direct-visual video-understanding agent. You inspect raw video frames directly in your conversation history and answer multiple-choice questions using visual evidence and temporal reasoning.

# Workflow
At each round, examine all acquired video frames shown so far, identify any missing visual detail, and choose ONE next tool call (`skim`, `focus`, or `answer`) with exact start_time/end_time in seconds.
- Do NOT output the final answer letter in your thought unless you are ready to call `answer`.
- You must finish when reaching the maximum number of steps.

# Toolkit
- `skim(query, start_time, end_time)`: Sample {skim_num_frames} raw frames uniformly from [start_time, end_time] and view them directly.
- `focus(query, start_time, end_time)`: Sample {focus_num_frames} high-resolution raw frames densely from a short clip (<= {focus_max_span}s) and view them directly.
- `answer()`: Finish evidence gathering and output the final multiple-choice answer.

# Rules
- Call ONLY ONE tool per turn.
- Do not repeat the exact same timestamp range if it already lacks decisive detail.
- Be concise in your Thought (3-5 sentences)."""


def _thin_image_history(history_entries, max_images):
    """Ensure total images across multimodal history stay <= max_images by thinning oldest overview frames first."""
    total = sum(len(e["images"]) for e in history_entries)
    if total <= max_images:
        return history_entries, 0
    pruned = 0
    entries = [dict(e, images=list(e["images"]), timestamps=list(e["timestamps"])) for e in history_entries]
    # Thin oldest entries first, always keeping at least 4 frames in the earliest overview if possible
    for idx in range(len(entries) - 1):
        excess = sum(len(e["images"]) for e in entries) - max_images
        if excess <= 0:
            break
        cur_n = len(entries[idx]["images"])
        min_keep = 4 if idx == 0 else 6
        can_drop = max(0, cur_n - min_keep)
        drop = min(excess, can_drop)
        if drop > 0:
            keep_n = cur_n - drop
            keep_idx = sorted(set(int(round(x)) for x in np.linspace(0, cur_n - 1, keep_n)))
            entries[idx]["images"] = [entries[idx]["images"][i] for i in keep_idx]
            entries[idx]["timestamps"] = [entries[idx]["timestamps"][i] for i in keep_idx]
            pruned += cur_n - len(keep_idx)
    # Hard safety cap if still over max_images
    while sum(len(e["images"]) for e in entries) > max_images and len(entries) > 1:
        excess = sum(len(e["images"]) for e in entries) - max_images
        for idx in range(len(entries)):
            if excess <= 0:
                break
            if len(entries[idx]["images"]) > 2:
                entries[idx]["images"] = entries[idx]["images"][::2]
                entries[idx]["timestamps"] = entries[idx]["timestamps"][::2]
                pruned += 1
                excess = sum(len(e["images"]) for e in entries) - max_images
    return entries, pruned


class ClockSplitRunner(VideoSeekLaveRunner):
    """Equal-budget 50/50 geometric split-budget control (ZERO LAVE visual-state receiver)."""

    def _run_uniform_bank_qa(self, bank, question_text, valid_letters, trajectory_brief, planner_reasoning=""):
        """Zero-pixel verification over 8 uniformly spaced bank frames (no LAVE lift ranking)."""
        if not bank:
            return dict(answer=None, raw_text="", selected_timestamps=[], num_frames=0, visual_tokens=0, visual_forwards=0, seconds=0.0)
        sorted_bank = sorted(bank, key=lambda e: float(e["timestamp_s"]))
        k = min(self.cfg.latent_qa_frames, len(sorted_bank))
        idxs = [int(round(x)) for x in np.linspace(0, len(sorted_bank) - 1, k)]
        seen = set()
        picked = []
        for i in idxs:
            if i not in seen:
                seen.add(i)
                picked.append(sorted_bank[i])
        sel_slots = [e["slot"] for e in picked]
        sel_grids = [e["grid"] for e in picked]
        sel_meta = [dict(timestamp_s=e["timestamp_s"], tool=e["tool"]) for e in picked]
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
            raise RuntimeError("Uniform bank QA violated zero-vision invariant!")
        return dict(
            answer=extract_choice_letter(gen["text"], valid_letters),
            raw_text=gen["text"].strip(),
            selected_timestamps=[m["timestamp_s"] for m in sel_meta],
            selected_tools=[m["tool"] for m in sel_meta],
            num_frames=gen["num_frames"],
            visual_tokens=gen["visual_tokens"],
            visual_forwards=0,
            seconds=gen["seconds"],
        )

    def run_clock_split(self, case, shared_overview, *, tool_cache=None):
        """Run multi-step VideoSeek with deterministic 50/50 geometric Clock-Split (0 receiver calls)."""
        t_start = perf_counter()
        question_text, valid_letters = format_mc_question(case["question"], case["options"])
        duration = float(case["duration"])
        video_path = case["video_path"]

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Video Duration: {duration:.1f}s\n\nQuestion:\n{question_text}"},
        ]
        bank = list(shared_overview["bank_entries"])
        steps_log = []
        visited_windows = []
        seen_calls = set()
        total_frames = shared_overview["frames"]
        total_vision_calls = shared_overview["vision_calls"]
        visual_tool_calls = 1
        reroute_count = 0

        ov_thought = (
            "No prior visual observations are available yet. "
            "I will call `overview` to get a structured timeline of the entire video."
        )
        messages.append({"role": "user", "content": f"Step [1 / {self.cfg.max_steps}]: Follow the Thinking Policy and plan the next action."})
        messages.append({"role": "assistant", "content": ov_thought})
        messages.append({
            "role": "user",
            "content": f"Observation from `overview()` (0.0s - {duration:.1f}s, {shared_overview['frames']} frames):\n{shared_overview['text']}",
        })
        steps_log.append(dict(step_id=1, tool="overview", arguments={}, frames=shared_overview["frames"], rerouted=False, seconds=shared_overview["seconds"]))

        final_text_answer = None
        final_text_raw = ""
        step4_text_answer = None
        step4_frames = None
        finish_reason = "reach_max_steps"

        for step_id in range(2, self.cfg.max_steps + 1):
            messages.append({
                "role": "user",
                "content": (
                    f"Step [{step_id} / {self.cfg.max_steps}]: "
                    "Briefly summarize what evidence you have, identify any missing visual detail, "
                    "and state ONE next tool call (`skim`, `focus`, or `answer`) with exact start_time/end_time in seconds."
                ),
            })
            thought_out = self.engine.generate_text(messages, max_tokens=self.cfg.thought_max_tokens)
            thought = strip_think_tags(thought_out["text"]) or "Need more visual evidence."
            messages.append({"role": "assistant", "content": thought})

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
            parse_out = self.engine.generate_text(parse_msgs, max_tokens=self.cfg.parser_max_tokens, tools=self.tools)
            action = parse_tool_call(parse_out["text"]) or parse_tool_call(thought)
            if action is None:
                if step_id == self.cfg.max_steps:
                    action = dict(name="answer", arguments={}, parse_mode="max_step_fallback")
                else:
                    steps_log.append(dict(step_id=step_id, tool="invalid", raw=parse_out["text"][:200], frames=0, rerouted=False))
                    messages.append({"role": "user", "content": "Invalid tool call. You MUST invoke exactly one tool: skim, focus, or answer."})
                    continue

            tname = action["name"]
            args = dict(action.get("arguments") or {})

            if tname == "answer":
                ans_msgs = list(messages) + [{
                    "role": "user",
                    "content": f"Question:\n{question_text}\n\nRespond with ONLY the single best option letter ({', '.join(valid_letters)}).",
                }]
                ans_out = self.engine.generate_text(ans_msgs, max_tokens=self.cfg.answer_max_tokens)
                final_text_raw = ans_out["text"].strip()
                final_text_answer = extract_choice_letter(final_text_raw, valid_letters)
                finish_reason = "answer_tool"
                steps_log.append(dict(step_id=step_id, tool="answer", arguments={}, frames=0, rerouted=False, seconds=ans_out["seconds"]))
                break

            if tname == "overview":
                messages.append({
                    "role": "user",
                    "content": "Observation: `overview` was already executed at Step 1. Please use `skim` or `focus` on a specific time range, or call `answer`.",
                })
                steps_log.append(dict(step_id=step_id, tool="overview_skip", arguments={}, frames=0, rerouted=False))
                continue

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
            seen_calls.add((tname, round(p_start, 0), round(p_end, 0)))

            n_frames = self.cfg.focus_frames if tname == "focus" else self.cfg.skim_frames
            geo_peaks = choose_geometric_clock_peak(duration, p_start, p_end, visited_windows)
            plan = plan_split_budget_timestamps(tname, p_start, p_end, duration, n_frames, geo_peaks)
            ts_req = plan["timestamps"]
            rerouted = plan["rerouted"]
            if rerouted:
                reroute_count += 1
                lw_str = ", ".join(f"{lo:.1f}s-{hi:.1f}s" for lo, hi in plan["latent_windows"])
                window_note = f"Planner {p_start:.1f}s-{p_end:.1f}s + Clock-Split {lw_str}"
            else:
                window_note = f"{p_start:.1f}s-{p_end:.1f}s"

            tool_out = self._exec_skim_or_focus(
                tname, video_path, duration, query, ts_req, step_id, window_note, tool_cache=tool_cache
            )
            bank.extend(tool_out["bank_entries"])
            total_frames += tool_out["frames"]
            total_vision_calls += tool_out["vision_calls"]
            visual_tool_calls += 1
            visited_windows.append((p_start, p_end))
            if plan["latent_windows"]:
                visited_windows.extend(plan["latent_windows"])

            messages.append({
                "role": "user",
                "content": f"Observation from `{tname}` ({window_note}, query={query!r}):\n{tool_out['text']}",
            })
            steps_log.append(dict(
                step_id=step_id,
                tool=tname,
                arguments=dict(query=query, start_time=round(p_start, 1), end_time=round(p_end, 1)),
                window_note=window_note,
                frames=tool_out["frames"],
                rerouted=rerouted,
                clock_peaks=geo_peaks,
                seconds=tool_out["seconds"],
            ))

            if step_id == 4 and step4_text_answer is None and self.cfg.max_steps > 4:
                s4_msgs = list(messages) + [{
                    "role": "user",
                    "content": f"You have reached the maximum number of steps.\nQuestion:\n{question_text}\n\nRespond with ONLY the single best option letter ({', '.join(valid_letters)}).",
                }]
                s4_out = self.engine.generate_text(s4_msgs, max_tokens=self.cfg.answer_max_tokens)
                step4_text_answer = extract_choice_letter(s4_out["text"].strip(), valid_letters)
                step4_frames = total_frames

        if final_text_answer is None:
            ans_msgs = list(messages) + [{
                "role": "user",
                "content": f"You have reached the maximum number of steps.\nQuestion:\n{question_text}\n\nRespond with ONLY the single best option letter ({', '.join(valid_letters)}).",
            }]
            ans_out = self.engine.generate_text(ans_msgs, max_tokens=self.cfg.answer_max_tokens)
            final_text_raw = ans_out["text"].strip()
            final_text_answer = extract_choice_letter(final_text_raw, valid_letters)

        if step4_text_answer is None:
            step4_text_answer = final_text_answer
            step4_frames = total_frames

        traj_brief = "\n\n".join(
            m["content"][:1200] for m in messages if m["role"] == "user" and m["content"].startswith("Observation")
        )
        planner_thoughts = "\n\n".join(m["content"][:600] for m in messages if m["role"] == "assistant")
        uniform_qa = self._run_uniform_bank_qa(bank, question_text, valid_letters, traj_brief, planner_thoughts)

        return dict(
            finish_reason=finish_reason,
            num_steps=len(steps_log),
            visual_tool_calls=visual_tool_calls,
            total_frames=total_frames,
            step4_frames=step4_frames,
            total_vision_calls=total_vision_calls,
            reroute_count=reroute_count,
            text_answer=final_text_answer,
            step4_text_answer=step4_text_answer,
            text_answer_raw=final_text_raw,
            uniform_qa=uniform_qa,
            steps=steps_log,
            elapsed_seconds=round(perf_counter() - t_start, 3),
        )


class VideoSeekImageRunner:
    """Direct raw-pixel interleaved multi-turn agent (VideoSeek-Image baseline, no observer captions)."""

    def __init__(self, engine, config: AgentConfig | None = None, *, max_context_frames=64):
        self.engine = engine
        self.cfg = config or AgentConfig()
        self.max_context_frames = int(max_context_frames)
        self.tools = build_videoseek_tools(
            overview_frames=self.cfg.overview_frames,
            skim_frames=self.cfg.skim_frames,
            focus_frames=self.cfg.focus_frames,
            focus_max_span=int(self.cfg.focus_max_span),
        )
        self.system_prompt = VIDEOSEEK_IMAGE_SYSTEM_PROMPT.format(
            skim_num_frames=self.cfg.skim_frames,
            focus_num_frames=self.cfg.focus_frames,
            focus_max_span=int(self.cfg.focus_max_span),
        )

    def _build_mm_messages(self, duration, question_text, visual_entries, thought_history, tail_instruction):
        """Build multi-turn multimodal prompt carrying all acquired raw visual frames directly."""
        kept_entries, pruned = _thin_image_history(visual_entries, self.max_context_frames)
        pil_images = []
        messages = [{"role": "system", "content": self.system_prompt}]

        # First user message holds the question + Step-1 overview raw frames
        first = kept_entries[0]
        c0 = [
            {
                "type": "text",
                "text": (
                    f"Video Duration: {duration:.1f}s\n\n"
                    f"Question:\n{question_text}\n\n"
                    f"Step 1 `overview` ({len(first['images'])} raw frames across 0.0s - {duration:.1f}s):\n"
                ),
            }
        ]
        for img, ts in zip(first["images"], first["timestamps"]):
            c0.append({"type": "text", "text": f"{ts:.1f}s:"})
            c0.append({"type": "image"})
            pil_images.append(img)
        messages.append({"role": "user", "content": c0})

        # Interleave previous thoughts and subsequent raw clip frames
        for i, th in enumerate(thought_history):
            messages.append({"role": "assistant", "content": th})
            if i + 1 < len(kept_entries):
                ent = kept_entries[i + 1]
                ci = [
                    {
                        "type": "text",
                        "text": f"Returned raw frames from `{ent['tool']}` ({ent['note']}, {len(ent['images'])} frames):\n",
                    }
                ]
                for img, ts in zip(ent["images"], ent["timestamps"]):
                    ci.append({"type": "text", "text": f"{ts:.1f}s:"})
                    ci.append({"type": "image"})
                    pil_images.append(img)
                messages.append({"role": "user", "content": ci})

        # Append current turn instruction to the last user message or as a fresh user turn
        if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
            messages[-1]["content"].append({"type": "text", "text": "\n" + tail_instruction})
        else:
            messages.append({"role": "user", "content": tail_instruction})
        return messages, pil_images, pruned

    def run_case_standalone(self, case):
        """Run standalone direct-visual VideoSeek-Image (zero shared overview/cache)."""
        t_start = perf_counter()
        question_text, valid_letters = format_mc_question(case["question"], case["options"])
        duration = float(case["duration"])
        video_path = case["video_path"]

        # Step 1: decode raw overview frames directly (no captioner call!)
        ov_req = uniform_timestamps(0.0, duration, self.cfg.overview_frames, duration)
        ov_imgs, ov_ts = decode_frames_pyav(
            video_path, ov_req, short_side=self.cfg.overview_short_side, max_pixels=self.cfg.focus_max_pixels
        )
        visual_entries = [dict(tool="overview", note=f"0.0s-{duration:.1f}s", images=ov_imgs, timestamps=ov_ts)]
        thought_history = []
        seen_calls = set()
        steps_log = [dict(step_id=1, tool="overview_raw", frames=len(ov_imgs), seconds=0.0)]

        total_frames_acquired = len(ov_imgs)
        total_frames_fed_vlm = 0
        total_prompt_tokens = 0
        total_output_tokens = 0
        total_pruned = 0
        vlm_turns = 0
        visual_tool_calls = 1

        final_answer = None
        final_raw = ""
        step4_answer = None
        step4_frames = None
        finish_reason = "reach_max_steps"

        for step_id in range(2, self.cfg.max_steps + 1):
            instr = (
                f"Step [{step_id} / {self.cfg.max_steps}]: "
                "Briefly summarize the visual evidence seen in the raw frames above, identify any missing visual detail, "
                "and state ONE next tool call (`skim`, `focus`, or `answer`) with exact start_time/end_time in seconds."
            )
            mm_msgs, pil_imgs, pruned = self._build_mm_messages(duration, question_text, visual_entries, thought_history, instr)
            total_pruned += pruned
            th_out = self.engine.generate_vlm(mm_msgs, pil_imgs, max_tokens=self.cfg.thought_max_tokens)
            vlm_turns += 1
            total_frames_fed_vlm += len(pil_imgs)
            total_prompt_tokens += int(th_out.get("prompt_tokens", 0))
            total_output_tokens += int(th_out.get("output_tokens", 0))
            thought = strip_think_tags(th_out["text"]) or "Need more visual evidence."

            # Action parser (same deterministic parser as VideoSeek)
            parse_msgs = [
                {"role": "system", "content": ACTION_PARSER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Video Duration: {duration:.1f}s\n\n"
                        f"Question:\n{question_text}\n\n"
                        f"Planner Thought:\n{thought}\n\n"
                        "Return exactly ONE function call."
                    ),
                },
            ]
            parse_out = self.engine.generate_text(parse_msgs, max_tokens=self.cfg.parser_max_tokens, tools=self.tools)
            total_prompt_tokens += int(parse_out.get("prompt_tokens", 0))
            total_output_tokens += int(parse_out.get("output_tokens", 0))
            action = parse_tool_call(parse_out["text"]) or parse_tool_call(thought)
            if action is None:
                if step_id == self.cfg.max_steps:
                    action = dict(name="answer", arguments={}, parse_mode="max_step_fallback")
                else:
                    thought_history.append(thought)
                    steps_log.append(dict(step_id=step_id, tool="invalid", frames=0, seconds=th_out["seconds"]))
                    continue

            tname = action["name"]
            args = dict(action.get("arguments") or {})

            if tname == "answer":
                thought_history.append(thought)
                ans_instr = (
                    f"Question:\n{question_text}\n\n"
                    f"Based directly on all the video frames shown above, respond with ONLY the single best option letter ({', '.join(valid_letters)})."
                )
                ans_msgs, ans_imgs, pr2 = self._build_mm_messages(duration, question_text, visual_entries, thought_history, ans_instr)
                total_pruned += pr2
                ans_out = self.engine.generate_vlm(ans_msgs, ans_imgs, max_tokens=self.cfg.answer_max_tokens)
                vlm_turns += 1
                total_frames_fed_vlm += len(ans_imgs)
                total_prompt_tokens += int(ans_out.get("prompt_tokens", 0))
                total_output_tokens += int(ans_out.get("output_tokens", 0))
                final_raw = ans_out["text"].strip()
                final_answer = extract_choice_letter(final_raw, valid_letters)
                finish_reason = "answer_tool"
                steps_log.append(dict(step_id=step_id, tool="answer", frames=0, context_frames=len(ans_imgs), seconds=round(th_out["seconds"] + ans_out["seconds"], 3)))
                break

            if tname == "overview":
                thought_history.append(thought)
                steps_log.append(dict(step_id=step_id, tool="overview_skip", frames=0, seconds=th_out["seconds"]))
                continue

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
            sig = (tname, round(p_start, 0), round(p_end, 0))
            if sig in seen_calls:
                shift = 12.0 if p_end + 12.0 <= duration else -12.0
                p_start = max(0.0, min(duration - 2.0, p_start + shift))
                p_end = max(p_start + 2.0, min(duration, p_end + shift))
            seen_calls.add(sig)

            n_frames = self.cfg.focus_frames if tname == "focus" else self.cfg.skim_frames
            ts_req = uniform_timestamps(p_start, p_end, n_frames, duration)
            if tname == "skim":
                imgs, ts_act = decode_frames_pyav(video_path, ts_req, short_side=self.cfg.skim_short_side, max_pixels=self.cfg.focus_max_pixels)
            else:
                imgs, ts_act = decode_frames_pyav(video_path, ts_req, short_side=None, max_pixels=self.cfg.focus_max_pixels)

            thought_history.append(thought)
            visual_entries.append(dict(tool=tname, note=f"{p_start:.1f}s-{p_end:.1f}s", images=imgs, timestamps=ts_act))
            total_frames_acquired += len(imgs)
            visual_tool_calls += 1
            steps_log.append(dict(
                step_id=step_id,
                tool=tname,
                arguments=dict(start_time=round(p_start, 1), end_time=round(p_end, 1)),
                frames=len(imgs),
                context_frames=len(pil_imgs),
                seconds=th_out["seconds"],
            ))

            if step_id == 4 and step4_answer is None and self.cfg.max_steps > 4:
                s4_instr = (
                    f"You have reached the maximum number of steps.\nQuestion:\n{question_text}\n\n"
                    f"Respond with ONLY the single best option letter ({', '.join(valid_letters)})."
                )
                s4_msgs, s4_imgs, _ = self._build_mm_messages(duration, question_text, visual_entries, thought_history, s4_instr)
                s4_out = self.engine.generate_vlm(s4_msgs, s4_imgs, max_tokens=self.cfg.answer_max_tokens)
                step4_answer = extract_choice_letter(s4_out["text"].strip(), valid_letters)
                step4_frames = total_frames_acquired

        if final_answer is None:
            ans_instr = (
                f"You have reached the maximum number of steps.\nQuestion:\n{question_text}\n\n"
                f"Respond with ONLY the single best option letter ({', '.join(valid_letters)})."
            )
            ans_msgs, ans_imgs, pr3 = self._build_mm_messages(duration, question_text, visual_entries, thought_history, ans_instr)
            total_pruned += pr3
            ans_out = self.engine.generate_vlm(ans_msgs, ans_imgs, max_tokens=self.cfg.answer_max_tokens)
            vlm_turns += 1
            total_frames_fed_vlm += len(ans_imgs)
            total_prompt_tokens += int(ans_out.get("prompt_tokens", 0))
            total_output_tokens += int(ans_out.get("output_tokens", 0))
            final_raw = ans_out["text"].strip()
            final_answer = extract_choice_letter(final_raw, valid_letters)

        if step4_answer is None:
            step4_answer = final_answer
            step4_frames = total_frames_acquired

        return dict(
            finish_reason=finish_reason,
            num_steps=len(steps_log),
            visual_tool_calls=visual_tool_calls,
            vlm_turns=vlm_turns,
            total_frames=total_frames_acquired,
            step4_frames=step4_frames,
            total_frames_fed_vlm=total_frames_fed_vlm,
            pruned_frames=total_pruned,
            total_prompt_tokens=total_prompt_tokens,
            total_output_tokens=total_output_tokens,
            answer=final_answer,
            step4_answer=step4_answer,
            raw_text=final_raw,
            steps=steps_log,
            elapsed_seconds=round(perf_counter() - t_start, 3),
        )


def run_standalone_unshared_latency(engine, cfg, case, ordinal):
    """Measure completely unshared, end-to-end wall-clock latency for Vanilla vs LAVE (Route & Full)."""
    runner = VideoSeekLaveRunner(engine, cfg)

    # 1. Standalone Vanilla VideoSeek (fresh overview, zero tool_cache)
    t0 = perf_counter()
    ov_v = runner.run_shared_overview(case["video_path"], float(case["duration"]))
    van = runner._run_trajectory(case, ov_v, use_lave_routing=False, seed_base=ordinal + 10000, tool_cache=None)
    t_van_total = perf_counter() - t0
    t_van_text_only = round(t_van_total - float(van["latent_qa"]["seconds"]), 3)

    # 2. Standalone VideoSeek + LAVE (fresh overview, zero tool_cache)
    t1 = perf_counter()
    ov_l = runner.run_shared_overview(case["video_path"], float(case["duration"]))
    lav = runner._run_trajectory(case, ov_l, use_lave_routing=True, seed_base=ordinal + 20000, tool_cache=None)
    t_lav_full = perf_counter() - t1
    t_lav_route_only = round(t_lav_full - float(lav["latent_qa"]["seconds"]), 3)

    return dict(
        vanilla_text_s=t_van_text_only,
        vanilla_latent_qa_s=round(t_van_total, 3),
        vanilla_frames=van["total_frames"],
        vanilla_visual_calls=van["visual_tool_calls"],
        lave_route_text_s=t_lav_route_only,
        lave_full_s=round(t_lav_full, 3),
        lave_receiver_s=lav["receiver_seconds"],
        lave_latent_qa_s=lav["latent_qa"]["seconds"],
        lave_frames=lav["total_frames"],
        lave_visual_calls=lav["visual_tool_calls"],
    )
