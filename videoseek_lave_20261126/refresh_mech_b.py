"""Zero-waste Option-B Mechanism-B refresher for already-committed cases (preserves trajectories & audit trail)."""

import json
from pathlib import Path
from time import perf_counter

from .agent import (
    AgentConfig,
    _next_vlm_call_uid,
    decode_frames_pyav,
    extract_choice_letter,
)
from .holmes_inputs import write_json_canonical
from .prompts import (
    filter_nonempty_observations,
    format_mc_question,
    render_latent_readout_prompt,
)


def _capture_unique_frame_states(engine, cfg, video_path, pairs):
    """Decode and capture exact post-merger embeddings for unique (ts, tool) pairs in a single 1-token forward."""
    uniq = []
    seen = set()
    for ts, tool in pairs:
        key = (round(float(ts), 2), str(tool))
        if key not in seen:
            seen.add(key)
            uniq.append(key)

    images = []
    for ts, tool in uniq:
        if tool in ("overview", "skim"):
            imgs, _ = decode_frames_pyav(
                video_path, [ts], short_side=cfg.overview_short_side, max_pixels=cfg.focus_max_pixels
            )
        else:
            imgs, _ = decode_frames_pyav(
                video_path, [ts], short_side=None, max_pixels=cfg.focus_max_pixels
            )
        images.append(imgs[0])

    uid = _next_vlm_call_uid()
    content = [{"type": "text", "text": f"[BankCapture #{uid}] Frames:\n"}]
    for (ts, _), _img in zip(uniq, images):
        content.append({"type": "text", "text": f"{ts:.1f}s:"})
        content.append({"type": "image"})
    content.append({"type": "text", "text": "OK"})

    out = engine.generate_vlm_with_capture([{"role": "user", "content": content}], images, max_tokens=1)
    state_map = {}
    for i, key in enumerate(uniq):
        state_map[key] = (out["slots"][i], out["grids"][i])
    return state_map


def _build_clean_brief_from_record(rec, arm_key):
    blocks = []
    ov_txt = (rec.get("shared_overview", {}).get("text_preview") or "").strip()
    if ov_txt:
        blocks.append(f"Observation from `overview()`:\n{ov_txt}")
    for st in rec[arm_key]["steps"]:
        if st["tool"] in ("skim", "focus"):
            obs = (st.get("obs_preview") or "").strip()
            if obs:
                blocks.append(f"Observation from `{st['tool']}` ({st.get('window_note', '')}):\n{obs}")
    return filter_nonempty_observations(blocks)


def refresh_suite_mech_b(engine, cfg, cases, case_dir: Path, suite_name: str):
    """Refresh Mechanism-B readout on already-committed cases that predate option_b_clean (zero trajectory rerun)."""
    to_refresh = []
    for ordinal, case in enumerate(cases):
        cpath = case_dir / f"case_{ordinal:03d}.json"
        if not cpath.exists():
            continue
        rec = json.loads(cpath.read_bytes())
        if rec.get("lave", {}).get("latent_qa", {}).get("mech_b_version") == "option_b_clean":
            continue
        to_refresh.append((ordinal, case, cpath, rec))

    if not to_refresh:
        return 0

    print(
        json.dumps(
            dict(status="mech_b_refresh_start", suite=suite_name, count=len(to_refresh)),
            sort_keys=True,
        ),
        flush=True,
    )
    t_all = perf_counter()
    for idx, (ordinal, case, cpath, rec) in enumerate(to_refresh):
        question_text, valid_letters = format_mc_question(case["question"], case["options"])
        v_lq = rec["vanilla"]["latent_qa"]
        l_lq = rec["lave"]["latent_qa"]
        v_pairs = list(zip(v_lq["selected_timestamps"], v_lq["selected_tools"]))
        l_pairs = list(zip(l_lq["selected_timestamps"], l_lq["selected_tools"]))

        state_map = _capture_unique_frame_states(engine, cfg, case["video_path"], v_pairs + l_pairs)

        if "answers_option_a_orig" not in rec:
            rec["answers_option_a_orig"] = dict(rec["answers"])

        for arm_key, ans_clean_key, ans_vo_key, pairs in (
            ("vanilla", "vanilla_latent_qa", "vanilla_latent_visonly", v_pairs),
            ("lave", "lave_full", "lave_full_visonly", l_pairs),
        ):
            sel_slots = [state_map[(round(float(t), 2), str(tl))][0] for t, tl in pairs]
            sel_grids = [state_map[(round(float(t), 2), str(tl))][1] for t, tl in pairs]
            sel_meta = [dict(timestamp_s=float(t), tool=str(tl)) for t, tl in pairs]
            clean_brief = _build_clean_brief_from_record(rec, arm_key)

            p_clean = render_latent_readout_prompt(
                engine.tokenizer, question_text, valid_letters, sel_meta, clean_brief, visual_only=False
            )
            g_clean = engine.generate_from_visual_embeds(
                p_clean, sel_slots, sel_grids, max_tokens=cfg.answer_max_tokens
            )
            if g_clean["visual_forwards"] != 0:
                raise RuntimeError("Option-B clean readout violated zero-vision invariant!")
            ch_clean = extract_choice_letter(g_clean["text"], valid_letters)

            p_vo = render_latent_readout_prompt(
                engine.tokenizer, question_text, valid_letters, sel_meta, "", visual_only=True
            )
            g_vo = engine.generate_from_visual_embeds(
                p_vo, sel_slots, sel_grids, max_tokens=cfg.answer_max_tokens
            )
            if g_vo["visual_forwards"] != 0:
                raise RuntimeError("Option-B visual-only readout violated zero-vision invariant!")
            ch_vo = extract_choice_letter(g_vo["text"], valid_letters)

            lq = rec[arm_key]["latent_qa"]
            lq["answer_option_a_orig"] = lq.get("answer")
            lq["answer"] = ch_clean
            lq["raw_text"] = g_clean["text"].strip()
            lq["answer_visual_only"] = ch_vo
            lq["raw_text_visual_only"] = g_vo["text"].strip()
            lq["mech_b_version"] = "option_b_clean"
            lq["visual_forwards"] = 0
            rec["answers"][ans_clean_key] = ch_clean
            rec["answers"][ans_vo_key] = ch_vo

        write_json_canonical(cpath, rec)
        if (idx + 1) % 50 == 0 or (idx + 1) == len(to_refresh):
            print(
                json.dumps(
                    dict(
                        status="mech_b_refresh_progress",
                        suite=suite_name,
                        done=idx + 1,
                        total=len(to_refresh),
                        elapsed_s=round(perf_counter() - t_all, 1),
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )
    return len(to_refresh)
