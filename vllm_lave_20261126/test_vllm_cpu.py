"""Pure-CPU unit tests for vllm_lave_20261126 (no GPU or vLLM runtime required)."""

from types import SimpleNamespace
import unittest

from missing_fact_routing_20261124.target import target_from_question
from vllm_lave_20261126.engine import (
    IMAGE_PLACEHOLDER,
    VLLMLaveEngine,
    extract_yn_logits,
    render_receiver_vllm_prompt,
    split_captured_slots,
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False, tools=None):
        parts = [f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages]
        return "\n".join(parts) + "\n<assistant>"

    def encode(self, text, add_special_tokens=False):
        out = [ord(c) for c in text]
        return out

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeVisual:
    def __init__(self):
        self.spatial_merge_size = 2
        self._pre_hooks = []
        self._post_hooks = []
        self.calls = 0

    def register_forward_pre_hook(self, fn):
        self._pre_hooks.append(fn)
        return SimpleNamespace(remove=lambda: self._pre_hooks.remove(fn))

    def register_forward_hook(self, fn, with_kwargs=False):
        self._post_hooks.append(fn)
        return SimpleNamespace(remove=lambda: self._post_hooks.remove(fn))

    def call_visual(self, fake_out, fake_grid):
        self.calls += 1
        for fn in list(self._pre_hooks):
            fn(self, (), {})
        for fn in list(self._post_hooks):
            fn(self, (), {"grid_thw": fake_grid}, fake_out)
        return fake_out


class FakeLLM:
    def __init__(self, invoke_visual_on_receiver=False):
        self.model = SimpleNamespace(visual=FakeVisual())
        self.invoke_visual_on_receiver = invoke_visual_on_receiver
        self.last_requests = None

    def apply_model(self, fn):
        return [fn(self.model)]

    def generate(self, requests, sampling_params=None, use_tqdm=False):
        self.last_requests = requests
        if self.invoke_visual_on_receiver:
            self.model.visual.call_visual("boom", "boom")
        outs = []
        for idx, _ in enumerate(requests):
            y_id, n_id = sampling_params["allowed_token_ids"]
            lp = {y_id: FakeLogprob(1.5 + idx * 0.1), n_id: FakeLogprob(-0.5)}
            outs.append(SimpleNamespace(outputs=[SimpleNamespace(text="Y", token_ids=[y_id], finish_reason="stop", logprobs=[lp])]))
        return outs


class VLLMLaveCPUTest(unittest.TestCase):
    def test_render_receiver_vllm_prompt_single_placeholder(self):
        tok = FakeTokenizer()
        target = target_from_question("What is the price of the Jinhao 80 fountain pen?")
        prompt, (y_id, n_id) = render_receiver_vllm_prompt(tok, target, "find price", ["rep1", "rep2"])
        self.assertEqual(prompt.count(IMAGE_PLACEHOLDER), 1)
        self.assertEqual(prompt.count("<|image_pad|>"), 1)
        self.assertEqual((y_id, n_id), (ord("Y"), ord("N")))

    def test_render_rejects_special_token_injection(self):
        tok = FakeTokenizer()
        target = target_from_question("How many cars appear?")
        with self.assertRaises(ValueError):
            render_receiver_vllm_prompt(tok, target, "cars", ["bad <|image_pad|>", "ok"])

    def test_split_captured_slots_exact(self):
        # 2 frames of grid (1, 4, 4) with merge=2 -> 4 tokens each -> 8 rows total
        embeds = list(range(8))
        slots, grids = split_captured_slots(embeds, [(1, 4, 4), (1, 4, 4)], spatial_merge_size=2)
        self.assertEqual(len(slots), 2)
        self.assertEqual(slots[0], [0, 1, 2, 3])
        self.assertEqual(slots[1], [4, 5, 6, 7])
        self.assertEqual(grids, ((1, 4, 4), (1, 4, 4)))
        with self.assertRaises(ValueError):
            split_captured_slots(embeds[:-1], [(1, 4, 4), (1, 4, 4)], spatial_merge_size=2)

    def test_extract_yn_logits_exact_and_missing_check(self):
        lp = {89: FakeLogprob(2.25), 78: FakeLogprob(-1.75)}
        self.assertEqual(extract_yn_logits(lp, 89, 78), (2.25, -1.75))
        with self.assertRaises(ValueError):
            extract_yn_logits({89: FakeLogprob(2.25)}, 89, 78)
        with self.assertRaises(ValueError):
            extract_yn_logits({89: FakeLogprob(float("nan")), 78: FakeLogprob(0.0)}, 89, 78)

    def test_receiver_forbids_vision_calls(self):
        import sys
        import types

        # Provide minimal fake torch.tensor so this test runs cleanly even without local torch
        fake_torch = types.SimpleNamespace(long="long", tensor=lambda x, dtype=None: x)
        prev = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            llm_ok = FakeLLM(invoke_visual_on_receiver=False)
            eng_ok = VLLMLaveEngine("dummy", llm=llm_ok, tokenizer=FakeTokenizer())
            eng_ok._make_sampling_params = lambda **kw: kw
            res = eng_ok.score_receiver_slots("p", (1, 4, 4), ["s0", "s1"], 89, 78, batch_size=2)
            self.assertEqual(res["forwards"], 2)
            self.assertEqual(res["visual_forwards"], 0)
            self.assertEqual(res["scores"][0], (1.5, -0.5))
            self.assertEqual(llm_ok.model.visual.calls, 0)
            self.assertEqual(len(llm_ok.model.visual._pre_hooks), 0)

            llm_bad = FakeLLM(invoke_visual_on_receiver=True)
            eng_bad = VLLMLaveEngine("dummy", llm=llm_bad, tokenizer=FakeTokenizer())
            eng_bad._make_sampling_params = lambda **kw: kw
            with self.assertRaises(RuntimeError):
                eng_bad.score_receiver_slots("p", (1, 4, 4), ["s0"], 89, 78, batch_size=1)
            self.assertEqual(len(llm_bad.model.visual._pre_hooks), 0)
        finally:
            if prev is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = prev


if __name__ == "__main__":
    unittest.main()
