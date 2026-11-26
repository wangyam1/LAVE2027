"""Official VideoSeek prompts/tools plus LAVE single-video visual-state receiver & latent QA prompts."""

from collections.abc import Mapping


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


# Faithful to official jylins/videoseek config/prompts.yaml (parameterized for frame counts)
VIDEOSEEK_SYSTEM_PROMPT = """# Role
You are an efficient video-understanding agent that reasons like a careful human watcher. You answer multiple-choice questions using partial observations and the logical structure of the video (temporal order, causality), rather than exhaustively parsing every frame.

# Workflow
Basic workflow at each round: Thought -> Action -> Observation, repeated until sufficient evidence is gathered.
- Thought: Given the previous trajectory and the question, determine whether the collected evidence is sufficient to answer the question. If so, plan to call `answer`. If not, identify the missing visual evidence and plan the next single tool call (`overview`, `skim`, or `focus`) with specific timestamps.
- Do NOT provide the final answer letter in your thought unless you are ready to call `answer`.
- You must finish when reaching the maximum number of steps.

# Toolkit
## `overview`: whole-video summary
- When: No prior information is available or for global structure.
- How: Samples {overview_num_frames} frames uniformly from the entire video.

## `skim`: coarse scan of a long segment (> {skim_num_frames}s)
- When: To localize moments related to a query across a candidate time range.
- Parameters: `query` (concise visual question), `start_time` (seconds), `end_time` (seconds).
- Samples {skim_num_frames} frames uniformly from [start_time, end_time].

## `focus`: dense inspection of a short clip (<= {focus_max_span}s)
- When: To verify fine visual details (expressions, text, objects, actions) for a query.
- Parameters: `query` (concise visual question), `start_time` (seconds), `end_time` (seconds).
- Samples {focus_num_frames} frames densely from [start_time, end_time].

## `answer`: finish the trajectory and output the final answer
- When: Collected observations clearly support one option (or at final step).

# Rules
- Call ONLY ONE tool per turn.
- Start with `overview` if no observation has been gathered yet.
- If a previous `focus` or `skim` returned "No relevant content found." or lacked decisive detail, target a DIFFERENT time range rather than repeating the same timestamps.
- Be concise in your Thought (3-6 sentences)."""


ACTION_PARSER_PROMPT = (
    "You are a deterministic tool-call parser.\n"
    "Convert the planner thought into exactly ONE function call.\n"
    "Rules:\n"
    "1) Call exactly one tool from the provided tool schema: overview, skim, focus, or answer.\n"
    "2) The tool name and numeric start_time/end_time (in seconds) must match the thought.\n"
    "3) If the thought says evidence is sufficient or identifies the final option, call `answer`.\n"
    "4) Return ONLY the tool call, no extra explanation."
)


def build_videoseek_tools(overview_frames=32, skim_frames=16, focus_frames=16, focus_max_span=24):
    return [
        {
            "type": "function",
            "function": {
                "name": "overview",
                "description": f"Get a structured timestamped visual summary across the entire video ({overview_frames} frames).",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "skim",
                "description": (
                    f"Coarsely scan a video segment [start_time, end_time] (span >= {skim_frames}s) "
                    f"by sampling {skim_frames} frames to localize query-relevant moments."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concise visual question to check in this segment.",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Start time in seconds.",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "End time in seconds.",
                        },
                    },
                    "required": ["query", "start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "focus",
                "description": (
                    f"Densely inspect a short clip [start_time, end_time] (span <= {focus_max_span}s) "
                    f"with {focus_frames} high-resolution frames to verify fine visual details."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Concise visual question to verify in this clip.",
                        },
                        "start_time": {
                            "type": "number",
                            "description": "Start time in seconds.",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "End time in seconds.",
                        },
                    },
                    "required": ["query", "start_time", "end_time"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "answer",
                "description": "Finish evidence gathering and output the final multiple-choice answer.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        },
    ]


LAVE_RECEIVER_SYSTEM_PROMPT = (
    "You are a visual evidence router for a long-video reasoning agent. "
    "You receive the question, the agent's current search query, a brief summary of what has been observed so far, "
    "and ONE candidate visual frame state from an earlier scan. "
    "Judge whether this frame directly shows (or is immediately adjacent to) visual evidence—such as objects, people, "
    "actions, on-screen text, expressions, or causal clues—that is relevant to answering the query/question and worth "
    "inspecting more closely. Respond with a single letter: Y if relevant, N if not."
)


def render_lave_receiver_prompt(tokenizer, question_text, query, trajectory_brief):
    """Build exact single-slot receiver prompt and verify single-token (Y, N) verbalizers."""
    clean_q = question_text.replace("<|", "")
    clean_query = (query or "").replace("<|", "")
    clean_brief = (trajectory_brief or "None yet.").replace("<|", "")
    if len(clean_brief) > 2200:
        clean_brief = clean_brief[-2200:]

    user_msg = (
        f"Question & Options:\n{clean_q}\n\n"
        f"Current visual search query:\n{clean_query}\n\n"
        f"Summary of observations so far:\n{clean_brief}\n\n"
        f"Candidate visual frame state:\n{IMAGE_PLACEHOLDER}\n\n"
        "Does this frame contain or directly point to visual evidence relevant to the query/question? "
        "Reply strictly with Y or N."
    )
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": LAVE_RECEIVER_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count("<|image_pad|>") != 1:
        raise ValueError("Expected exactly one <|image_pad|> placeholder in receiver prompt")

    base_ids = tokenizer.encode(rendered, add_special_tokens=False)
    labels = []
    for marker in ("Y", "N"):
        full = tokenizer.encode(rendered + marker, add_special_tokens=False)
        if full[:-1] != base_ids or tokenizer.decode(full[-1:]) != marker:
            raise ValueError(f"Verbalizer {marker!r} did not append a single token")
        labels.append(full[-1])
    return rendered, tuple(labels)


def render_latent_readout_prompt(tokenizer, question_text, valid_letters, selected_meta, trajectory_brief):
    """Build zero-pixel multi-frame latent visual QA prompt over M cached post-merger states."""
    clean_q = question_text.replace("<|", "")
    clean_brief = (trajectory_brief or "").replace("<|", "")
    if len(clean_brief) > 3200:
        clean_brief = clean_brief[-3200:]

    frame_lines = []
    for item in selected_meta:
        ts = float(item["timestamp_s"])
        tool = item["tool"]
        frame_lines.append(f"[{ts:.1f}s | {tool}]\n{IMAGE_PLACEHOLDER}")
    frames_block = "\n".join(frame_lines)
    letters_str = ", ".join(valid_letters)

    user_msg = (
        f"Retrieved key visual frames from the video (in chronological order):\n"
        f"{frames_block}\n\n"
        f"Text observations collected during search:\n{clean_brief}\n\n"
        f"Question:\n{clean_q}\n\n"
        f"Carefully rely on the visual evidence in the frames above together with the timeline. "
        f"Respond with ONLY the single best option letter ({letters_str}) and nothing else."
    )
    rendered = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": (
                    "You are an expert multimodal video reasoning assistant. "
                    "Answer the multiple-choice question by directly inspecting the provided visual frames "
                    "and cross-checking with the collected observations. Output ONLY a single uppercase option letter."
                ),
            },
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count("<|image_pad|>") != len(selected_meta):
        raise ValueError(
            f"Expected {len(selected_meta)} image placeholders, got {rendered.count('<|image_pad|>')}"
        )
    return rendered


def format_mc_question(question, options):
    """Format a multiple-choice question with A/B/C/... options."""
    if isinstance(options, Mapping):
        keys = sorted(options.keys())
        lines = [f"({k}) {options[k]}" for k in keys]
        valid = tuple(keys)
    else:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        valid = tuple(letters[i] for i in range(len(options)))
        lines = [f"({letters[i]}) {opt}" for i, opt in enumerate(options)]
    return f"{question.strip()}\n" + "\n".join(lines), valid
