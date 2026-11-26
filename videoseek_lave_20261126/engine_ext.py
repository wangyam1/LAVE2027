"""vLLM 0.19.0 multi-step LAVE engine extensions:
1. General zero-replay visual-state capture for any VLM call (overview / skim / focus).
2. Mixed-resolution Visual-State Bank scoring with per-grid zero-baseline calibration and 0 vision calls.
3. Multi-frame zero-pixel visual answering directly from M cached post-merger image_embeds (0 vision calls).
"""

from collections import defaultdict
from time import perf_counter

from vllm_lave_20261126.engine import (
    VLLMLaveEngine,
    _install_forbid_vision,
    _install_source_capture,
    _pop_source_capture,
    _remove_forbid_vision,
    extract_yn_logits,
    split_captured_slots,
)


def concat_multi_image_embeds(slots, grids, *, spatial_merge_size=2):
    """Validate and concatenate M cached post-merger visual slots for a single vLLM request."""
    import torch

    if not slots or len(slots) != len(grids):
        raise ValueError(f"Mismatched slots ({len(slots)}) and grids ({len(grids)})")
    flat_parts = []
    grid_rows = []
    for idx, (slot, grid) in enumerate(zip(slots, grids)):
        t, h, w = (int(grid[0]), int(grid[1]), int(grid[2]))
        if t <= 0 or h <= 0 or w <= 0 or h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(f"Invalid grid at index {idx}: {(t, h, w)}")
        expected = t * (h // spatial_merge_size) * (w // spatial_merge_size)
        if slot.ndim != 2 or slot.shape[0] != expected:
            raise ValueError(
                f"Slot {idx} shape {tuple(slot.shape)} does not match grid {(t, h, w)} (expected {expected} rows)"
            )
        flat_parts.append(slot.contiguous())
        grid_rows.append([t, h, w])
    image_embeds = torch.cat(flat_parts, dim=0)
    image_grid_thw = torch.tensor(grid_rows, dtype=torch.long)
    return image_embeds, image_grid_thw


class MultiStepVLLMLaveEngine(VLLMLaveEngine):
    """VLLMLaveEngine extended for multi-step VideoSeek + Episodic Visual-State Bank."""

    def generate_vlm_with_capture(self, messages, pil_images, *, max_tokens=1024):
        """Run any VLM tool (overview / skim / focus) and capture its post-merger visual slots for free."""
        return self.generate_source_shared(messages, pil_images, max_tokens=max_tokens)

    def score_mixed_grid_slots(self, prompt_text, slots, grids, y_id, n_id, *, batch_size=36):
        """Score slots with potentially heterogeneous grid_thw using 0 vision calls.

        Also computes one calibrated zero-baseline per unique grid_thw so lift(i) = margin(slot_i) - margin(zero_grid_i)
        is exact even when overview/skim (256p) and focus (high-res) coexist in the same bank.
        """
        import torch

        if len(slots) != len(grids):
            raise ValueError("slots and grids length mismatch")
        if not slots:
            return dict(
                real_scores=[],
                zero_by_grid={},
                lifts=[],
                forwards=0,
                visual_forwards=0,
                seconds=0.0,
            )

        unique_grids = []
        zero_slots = []
        seen = {}
        for slot, grid in zip(slots, grids):
            gkey = tuple(int(v) for v in grid)
            if gkey not in seen:
                seen[gkey] = len(unique_grids)
                unique_grids.append(gkey)
                zero_slots.append(torch.zeros_like(slot))

        req_slots = list(zero_slots) + list(slots)
        req_grids = list(unique_grids) + [tuple(int(v) for v in g) for g in grids]

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
                        "image_grid_thw": torch.tensor([list(g)], dtype=torch.long),
                    }
                },
            }
            for slot, g in zip(req_slots, req_grids)
        ]

        scores = []
        self.llm.apply_model(_install_forbid_vision)
        t0 = perf_counter()
        try:
            for start in range(0, len(requests), batch_size):
                chunk = requests[start : start + batch_size]
                outputs = self.llm.generate(chunk, sampling_params=params, use_tqdm=False)
                for req_out in outputs:
                    step0 = req_out.outputs[0].logprobs[0]
                    scores.append(extract_yn_logits(step0, y_id, n_id))
        finally:
            self.llm.apply_model(_remove_forbid_vision)
        elapsed = perf_counter() - t0

        n_zero = len(unique_grids)
        zero_by_grid = {unique_grids[i]: scores[i] for i in range(n_zero)}
        real_scores = scores[n_zero:]
        lifts = []
        for sc, g in zip(real_scores, grids):
            gkey = tuple(int(v) for v in g)
            zy0, zn0 = zero_by_grid[gkey]
            lifts.append((sc[0] - sc[1]) - (zy0 - zn0))

        return dict(
            real_scores=real_scores,
            zero_by_grid=zero_by_grid,
            lifts=lifts,
            forwards=len(scores),
            visual_forwards=0,
            seconds=round(elapsed, 3),
        )

    def score_scrambled_control(self, prompt_text, slot, grid, y_id, n_id, *, seed=16423):
        """Pure single-video semantic control: channel-permuted + sign-flipped same-norm noise."""
        import torch

        rng = torch.Generator().manual_seed(int(seed))
        channels = torch.randperm(slot.shape[1], generator=rng)
        signs = (torch.randint(0, 2, (slot.shape[1],), generator=rng) * 2 - 1).to(dtype=slot.dtype)
        perturbed = (slot[:, channels] * signs).contiguous()
        res = self.score_receiver_slots(prompt_text, grid, (perturbed,), y_id, n_id, batch_size=1)
        return dict(
            score=res["scores"][0],
            forwards=1,
            visual_forwards=0,
            seconds=res["seconds"],
        )

    def generate_from_visual_embeds(self, prompt_text, slots, grids, *, max_tokens=32, spatial_merge_size=2):
        """Run multi-frame visual answering directly from M cached post-merger embeddings with 0 vision calls."""
        image_embeds, image_grid_thw = concat_multi_image_embeds(
            slots, grids, spatial_merge_size=spatial_merge_size
        )
        params = self._make_sampling_params(temperature=0.0, max_tokens=max_tokens)
        req = {
            "prompt": prompt_text,
            "multi_modal_data": {
                "image": {
                    "image_embeds": image_embeds,
                    "image_grid_thw": image_grid_thw,
                }
            },
        }
        self.llm.apply_model(_install_forbid_vision)
        t0 = perf_counter()
        try:
            outputs = self.llm.generate([req], sampling_params=params, use_tqdm=False)
        finally:
            self.llm.apply_model(_remove_forbid_vision)
        elapsed = perf_counter() - t0
        out = outputs[0].outputs[0]
        return dict(
            text=out.text,
            output_tokens=len(out.token_ids),
            capped=out.finish_reason == "length",
            num_frames=len(slots),
            visual_tokens=int(image_embeds.shape[0]),
            visual_forwards=0,
            seconds=round(elapsed, 3),
        )
