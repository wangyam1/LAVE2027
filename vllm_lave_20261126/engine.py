"""Single-engine vLLM runtime for training-free LAVE visual-state routing on Qwen3.5-9B.

Key design points (verified against vLLM 0.17.1 and 0.19.0 Qwen3_5ForConditionalGeneration):
1. Single shared model instance for Source, Planner, Receiver, Focus-1/2, and QA.
2. Zero-replay Source visual-state capture: installs a read-only forward hook on
   ``model.visual`` via ``LLM.apply_model`` during ``source_shared`` prefill to capture
   the 32 post-merger image embeddings without calling the vision encoder a second time.
3. Zero-pixel Receiver scoring with native Qwen3.5 M-RoPE: passes precomputed
   ``multi_modal_data={"image": {"image_embeds": slot, "image_grid_thw": [[1, h, w]]}}``,
   which natively bypasses ``model.visual`` while computing exact 3D M-RoPE coordinates,
   and verifies zero vision calls via a temporary forbidding hook on ``model.visual``.
4. Exact (logit_Y, logit_N) extraction: uses ``logprobs_mode="processed_logits"`` with
   ``SamplingParams(max_tokens=1, temperature=0.0, logprobs=2, allowed_token_ids=[Y, N])``
   so both raw FP32 logits are always returned even when one ranks outside top-20.
5. Hybrid Gated DeltaNet + Attention prefix caching: ``enable_prefix_caching=True`` with
   ``mamba_cache_mode="align"`` caches the ~1,200-token Receiver text prefix across slots.
"""

from collections.abc import Mapping
from math import isfinite
from time import perf_counter

from missing_fact_routing_20261124 import receiver


IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"


def render_receiver_vllm_prompt(tokenizer, target, query, reports, *, rule=receiver.RULE):
    """Build the exact receiver prompt with a single image placeholder for vLLM expansion."""
    values = tuple(
        target.get(k) if isinstance(target, Mapping) else getattr(target, k)
        for k in receiver.FIELDS
    )
    if isinstance(reports, str) or len(reports) != 2 or any(
        not isinstance(v, str) or "<|" in v for v in (*values, query, *reports)
    ):
        raise ValueError("Invalid receiver prompt data")
    question, entity, attribute, time_condition = values
    if not all(v.strip() for v in (question, entity, attribute)):
        raise ValueError("Question, entity, and attribute are required")

    message = (
        f"Original question:\n{question}\n\n"
        f"Entity: {entity}\nRequested attribute or relation: {attribute}\n"
        f"Time condition: {time_condition or 'Unspecified'}\n\n"
        f"Tool query:\n{query}\n\n"
        f"Source report 1:\n{reports[0]}\n\nSource report 2:\n{reports[1]}\n\n"
        f"Extra visual observation:\n{IMAGE_PLACEHOLDER}"
    )
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": rule}, {"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered.count("<|image_pad|>") != 1:
        raise ValueError("Expected exactly one <|image_pad|> placeholder for vLLM expansion")

    base_ids = tokenizer.encode(rendered, add_special_tokens=False)
    labels = []
    for marker in ("Y", "N"):
        full = tokenizer.encode(rendered + marker, add_special_tokens=False)
        if full[:-1] != base_ids or tokenizer.decode(full[-1:]) != marker:
            raise ValueError(f"Verbalizer {marker!r} must append exactly one token")
        labels.append(full[-1])
    return rendered, tuple(labels)


def split_captured_slots(image_embeds, grid_thw, *, spatial_merge_size=2):
    """Split flat concatenated post-merger visual embeddings into per-frame slot tensors."""
    if len(grid_thw) == 0:
        raise ValueError("Empty grid_thw in captured Source visual output")
    sizes = []
    grids = []
    for row in grid_thw:
        t, h, w = (int(row[0]), int(row[1]), int(row[2]))
        if t <= 0 or h <= 0 or w <= 0 or h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(f"Invalid image_grid_thw row: {(t, h, w)}")
        sizes.append(t * (h // spatial_merge_size) * (w // spatial_merge_size))
        grids.append((t, h, w))
    total = sum(sizes)
    if len(image_embeds) != total:
        raise ValueError(f"Captured embedding rows ({len(image_embeds)}) != expected grid tokens ({total})")
    slots = []
    offset = 0
    for n in sizes:
        slots.append(image_embeds[offset:offset + n])
        offset += n
    return tuple(slots), tuple(grids)


def extract_yn_logits(token_logprobs, y_id, n_id):
    """Extract exact finite (logit_Y, logit_N) floats from vLLM step-0 logprobs mapping."""
    if not isinstance(token_logprobs, Mapping) or y_id not in token_logprobs or n_id not in token_logprobs:
        raise ValueError("vLLM did not return both Y and N logits (check logprobs_mode='processed_logits')")
    vals = []
    for tid in (y_id, n_id):
        entry = token_logprobs[tid]
        score = float(entry.logprob if hasattr(entry, "logprob") else entry)
        if not isfinite(score):
            raise ValueError(f"Nonfinite vLLM receiver logit for token {tid}: {score}")
        vals.append(score)
    return (vals[0], vals[1])


def _install_source_capture(model):
    """Worker-side hook: capture post-merger visual embeddings (cache- & duplicate-safe)."""
    _pop_source_capture(model)
    model._lave_captured = []
    model._lave_embed_slots = []
    model._lave_vision_calls = 0

    def _on_visual(_module, args, kwargs, output):
        import torch

        model._lave_vision_calls += 1
        grid = kwargs.get("grid_thw")
        if grid is None and len(args) >= 2:
            grid = args[1]
        if grid is None:
            raise ValueError("Could not locate grid_thw in model.visual call")
        tensor = output[0] if isinstance(output, tuple) else output
        grid_t = (
            grid.detach().cpu()
            if isinstance(grid, torch.Tensor)
            else torch.as_tensor(grid, dtype=torch.long).cpu()
        )
        if grid_t.ndim == 1:
            grid_t = grid_t.reshape(-1, 3)
        model._lave_captured.append((tensor.detach().cpu(), grid_t))

    model._lave_hook = model.visual.register_forward_hook(_on_visual, with_kwargs=True)

    if hasattr(model, "embed_input_ids"):
        orig_embed = model.embed_input_ids
        model._lave_orig_embed = orig_embed

        def _wrapped_embed(input_ids, multimodal_embeddings=None, *args, **kwargs):
            if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
                if isinstance(multimodal_embeddings, (list, tuple)):
                    for item in multimodal_embeddings:
                        model._lave_embed_slots.append(item.detach().cpu())
                else:
                    model._lave_embed_slots.append(multimodal_embeddings.detach().cpu())
            return orig_embed(input_ids, multimodal_embeddings, *args, **kwargs)

        model.embed_input_ids = _wrapped_embed
    return True


def _pop_source_capture(model):
    """Worker-side cleanup: remove capture hooks and return CPU post-merger embeddings."""
    hook = getattr(model, "_lave_hook", None)
    if hook is not None:
        hook.remove()
        model._lave_hook = None
    orig_embed = getattr(model, "_lave_orig_embed", None)
    if orig_embed is not None:
        model.embed_input_ids = orig_embed
        model._lave_orig_embed = None
    captured = getattr(model, "_lave_captured", [])
    embed_slots = getattr(model, "_lave_embed_slots", [])
    vision_calls = int(getattr(model, "_lave_vision_calls", 0))
    model._lave_captured = []
    model._lave_embed_slots = []
    model._lave_vision_calls = 0
    merge = int(getattr(model.visual, "spatial_merge_size", 2))
    return captured, embed_slots, vision_calls, merge


def _install_forbid_vision(model):
    """Worker-side guard: forbid any invocation of model.visual during Receiver scoring."""
    if hasattr(model, "_lave_forbid_hook") and model._lave_forbid_hook is not None:
        model._lave_forbid_hook.remove()

    def _forbid(*_args, **_kwargs):
        raise RuntimeError("Receiver invoked model.visual despite precomputed image_embeds!")

    model._lave_forbid_hook = model.visual.register_forward_pre_hook(_forbid)
    return True


def _remove_forbid_vision(model):
    hook = getattr(model, "_lave_forbid_hook", None)
    if hook is not None:
        hook.remove()
        model._lave_forbid_hook = None
    return True


class VLLMLaveEngine:
    """Single-GPU vLLM engine supporting zero-replay LAVE capture and native image_embeds routing."""

    def __init__(
        self,
        model_path,
        *,
        max_model_len=20000,
        max_num_batched_tokens=16384,
        gpu_memory_utilization=0.50,
        max_num_seqs=36,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        enforce_eager=False,
        mm_processor_kwargs=None,
        limit_mm_per_prompt=None,
        llm=None,
        tokenizer=None,
    ):
        self.model_path = str(model_path)
        self.mm_processor_kwargs = mm_processor_kwargs
        self._image_processor = None
        if llm is not None:
            self.llm = llm
            self.tokenizer = tokenizer
            return

        from vllm import LLM

        kwargs = dict(
            model=self.model_path,
            dtype="bfloat16",
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            limit_mm_per_prompt=limit_mm_per_prompt or {"image": 32},
            enable_mm_embeds=True,
            mm_processor_cache_gb=0,
            enable_prefix_caching=enable_prefix_caching,
            logprobs_mode="processed_logits",
            enforce_eager=enforce_eager,
            trust_remote_code=False,
        )
        if mamba_cache_mode is not None:
            kwargs["mamba_cache_mode"] = mamba_cache_mode
        if mm_processor_kwargs is not None:
            kwargs["mm_processor_kwargs"] = mm_processor_kwargs
        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _compute_image_grids(self, pil_images):
        if self._image_processor is None:
            from transformers import AutoImageProcessor

            kw = dict(local_files_only=True)
            if self.mm_processor_kwargs:
                kw.update(self.mm_processor_kwargs)
            self._image_processor = AutoImageProcessor.from_pretrained(self.model_path, **kw)
        out = self._image_processor(images=list(pil_images), return_tensors="pt")
        return tuple(tuple(int(v) for v in row) for row in out["image_grid_thw"].tolist())

    def _render_chat(self, messages, *, tools=None, enable_thinking=False):
        template = dict(tokenize=False, add_generation_prompt=True, enable_thinking=bool(enable_thinking))
        if tools is not None:
            template["tools"] = tools
        return self.tokenizer.apply_chat_template(messages, **template)

    def _make_sampling_params(self, **kwargs):
        from vllm import SamplingParams

        return SamplingParams(**kwargs)

    def generate_source_shared(self, messages, pil_images, *, max_tokens=2048):
        """Run source_shared once and capture its 32 post-merger image embeddings without replay."""
        prompt = self._render_chat(messages)
        params = self._make_sampling_params(temperature=0.0, max_tokens=max_tokens)
        self.llm.apply_model(_install_source_capture)
        t0 = perf_counter()
        try:
            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": {"image": list(pil_images)}}],
                sampling_params=params,
                use_tqdm=False,
            )
        finally:
            results = self.llm.apply_model(_pop_source_capture)
        elapsed = perf_counter() - t0

        captured, embed_slots, vision_calls, merge = results[0]
        import torch

        # Prefer embed_input_ids multimodal_embeddings (preserves exact order even under encoder cache or duplicate frames)
        if embed_slots:
            if len(embed_slots) == len(pil_images):
                slots = tuple(embed_slots)
                if captured:
                    all_grids = torch.cat([c[1] for c in captured], dim=0).tolist()
                    if len(all_grids) == len(pil_images):
                        grids = tuple(tuple(int(v) for v in row) for row in all_grids)
                    else:
                        grids = self._compute_image_grids(pil_images)
                else:
                    grids = self._compute_image_grids(pil_images)
            else:
                flat = torch.cat(embed_slots, dim=0)
                grids = self._compute_image_grids(pil_images)
                slots, grids = split_captured_slots(flat, grids, spatial_merge_size=merge)
        elif captured:
            all_embeds = torch.cat([c[0] for c in captured], dim=0)
            all_grids = torch.cat([c[1] for c in captured], dim=0)
            slots, grids = split_captured_slots(all_embeds, all_grids.tolist(), spatial_merge_size=merge)
        else:
            raise RuntimeError("No visual prefill captured during source_shared")

        if len(slots) != len(pil_images) or len(grids) != len(pil_images):
            raise RuntimeError(f"Expected {len(pil_images)} captured slots/grids, got {len(slots)}/{len(grids)}")
        for s, (t, h, w) in zip(slots, grids):
            expected_tok = t * (h // merge) * (w // merge)
            if len(s) != expected_tok:
                raise RuntimeError(f"Captured slot rows ({len(s)}) != expected grid tokens ({expected_tok})")
        out = outputs[0].outputs[0]
        return dict(
            text=out.text,
            output_tokens=len(out.token_ids),
            capped=out.finish_reason == "length",
            slots=slots,
            grids=grids,
            vision_chunks=len(captured),
            vision_calls=vision_calls,
            seconds=round(elapsed, 3),
        )

    def generate_vlm(self, messages, pil_images, *, max_tokens=2048, enable_thinking=False):
        """Run standard multimodal generation (source_query, Focus-1, Focus-2, VideoSeek-Image)."""
        return self.generate_vlm_batch(
            [(messages, pil_images)], max_tokens=max_tokens, enable_thinking=enable_thinking
        )[0]

    def generate_vlm_batch(self, items, *, max_tokens=2048, enable_thinking=False):
        """Batch multiple multimodal requests (messages, pil_images) into one vLLM generate call."""
        if not items:
            return []
        reqs = [
            {
                "prompt": self._render_chat(msgs, enable_thinking=enable_thinking),
                "multi_modal_data": {"image": list(imgs)},
            }
            for msgs, imgs in items
        ]
        params = self._make_sampling_params(temperature=0.0, max_tokens=max_tokens)
        t0 = perf_counter()
        outputs = self.llm.generate(reqs, sampling_params=params, use_tqdm=False)
        elapsed = perf_counter() - t0
        per_item = round(elapsed / len(items), 3)
        res = []
        for req_out in outputs:
            out = req_out.outputs[0]
            prompt_toks = len(getattr(req_out, "prompt_token_ids", None) or ())
            res.append(dict(
                text=out.text,
                prompt_tokens=prompt_toks,
                output_tokens=len(out.token_ids),
                capped=out.finish_reason == "length",
                seconds=per_item,
            ))
        return res

    def generate_text(self, messages, *, max_tokens=512, tools=None, enable_thinking=False):
        """Run pure-text generation (Planner, QA-1, QA-2); preserve <tool_call> tags when tools are supplied."""
        return self.generate_text_batch(
            [messages], max_tokens=max_tokens, tools=tools, enable_thinking=enable_thinking
        )[0]

    def generate_text_batch(self, messages_list, *, max_tokens=512, tools=None, enable_thinking=False):
        """Batch multiple pure-text requests into one vLLM generate call."""
        if not messages_list:
            return []
        reqs = [{"prompt": self._render_chat(m, tools=tools, enable_thinking=enable_thinking)} for m in messages_list]
        skip_special = tools is None
        params = self._make_sampling_params(
            temperature=0.0, max_tokens=max_tokens, skip_special_tokens=skip_special,
        )
        t0 = perf_counter()
        outputs = self.llm.generate(reqs, sampling_params=params, use_tqdm=False)
        elapsed = perf_counter() - t0
        per_item = round(elapsed / len(messages_list), 3)
        res = []
        for req_out in outputs:
            out = req_out.outputs[0]
            text = out.text
            if not skip_special:
                for eos_tok in ("<|im_end|>", "<|endoftext|>"):
                    if text.endswith(eos_tok):
                        text = text[:-len(eos_tok)]
            prompt_toks = len(getattr(req_out, "prompt_token_ids", None) or ())
            res.append(dict(
                text=text,
                prompt_tokens=prompt_toks,
                output_tokens=len(out.token_ids),
                capped=out.finish_reason == "length",
                seconds=per_item,
            ))
        return res

    def score_receiver_slots(self, prompt_text, grid_thw, slot_embeds_list, y_id, n_id, *, batch_size=33):
        """Score precomputed post-merger visual slots with 0 vision encoder calls and native M-RoPE."""
        import torch

        grid_tensor = torch.tensor([list(grid_thw)], dtype=torch.long)
        params = self._make_sampling_params(
            temperature=0.0,
            max_tokens=1,
            logprobs=2,
            allowed_token_ids=[int(y_id), int(n_id)],
        )
        requests = [
            {
                "prompt": prompt_text,
                "multi_modal_data": {
                    "image": {
                        "image_embeds": slot,
                        "image_grid_thw": grid_tensor,
                    }
                },
            }
            for slot in slot_embeds_list
        ]

        scores = []
        self.llm.apply_model(_install_forbid_vision)
        t0 = perf_counter()
        try:
            for start in range(0, len(requests), batch_size):
                chunk = requests[start:start + batch_size]
                outputs = self.llm.generate(chunk, sampling_params=params, use_tqdm=False)
                for req_out in outputs:
                    step0 = req_out.outputs[0].logprobs[0]
                    scores.append(extract_yn_logits(step0, y_id, n_id))
        finally:
            self.llm.apply_model(_remove_forbid_vision)
        elapsed = perf_counter() - t0
        return dict(scores=scores, forwards=len(scores), visual_forwards=0, seconds=round(elapsed, 3))

    def probe_states(self, target, query, reports, grid_thw, slots, *, batch_size=33):
        """Drop-in vLLM equivalent of receiver.probe_states (zero + 32 slots)."""
        import torch

        prompt_text, (y_id, n_id) = render_receiver_vllm_prompt(self.tokenizer, target, query, reports)
        nil = torch.zeros_like(slots[0])
        res = self.score_receiver_slots(
            prompt_text, grid_thw, (nil, *slots), y_id, n_id, batch_size=batch_size,
        )
        return dict(
            real=res["scores"][1:],
            zero=res["scores"][0],
            forwards=res["forwards"],
            visual_forwards=0,
            seconds=res["seconds"],
            prompt_text=prompt_text,
            labels=(y_id, n_id),
        )

    def probe_controls(self, seed_index, target, wrong_target, query, reports, grid_thw, slots, slot_idx):
        """Drop-in vLLM equivalent of receiver.probe_controls (scrambled, wrong_zero, wrong_real)."""
        import torch

        active = slots[slot_idx]
        rng = torch.Generator().manual_seed(16423 + int(seed_index))
        channels = torch.randperm(active.shape[1], generator=rng)
        signs = (torch.randint(0, 2, (active.shape[1],), generator=rng) * 2 - 1).to(dtype=active.dtype)
        perturbed = (active[:, channels] * signs).contiguous()
        nil = torch.zeros_like(active)

        actual_prompt, (y_id, n_id) = render_receiver_vllm_prompt(self.tokenizer, target, query, reports)
        wrong_prompt, (wy_id, wn_id) = render_receiver_vllm_prompt(self.tokenizer, wrong_target, query, reports)
        if (y_id, n_id) != (wy_id, wn_id):
            raise ValueError("Verbalizer IDs differed between actual and foil prompts")

        t0 = perf_counter()
        scram_res = self.score_receiver_slots(actual_prompt, grid_thw, (perturbed,), y_id, n_id, batch_size=1)
        wrong_res = self.score_receiver_slots(wrong_prompt, grid_thw, (nil, active), y_id, n_id, batch_size=2)
        return dict(
            scrambled=scram_res["scores"][0],
            wrong_zero=wrong_res["scores"][0],
            wrong=wrong_res["scores"][1],
            forwards=3,
            visual_forwards=0,
            seconds=round(perf_counter() - t0, 3),
        )
