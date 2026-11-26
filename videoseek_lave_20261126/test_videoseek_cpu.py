"""CPU unit tests for multi-step VideoSeek + LAVE (no GPU required)."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from videoseek_lave_20261126.agent import (
    extract_choice_letter,
    normalized_positive_entropy,
    parse_tool_call,
    plan_split_budget_timestamps,
    select_latent_peaks,
    select_top_bank_entries,
)
from videoseek_lave_20261126.engine_ext import concat_multi_image_embeds
from videoseek_lave_20261126.holmes_inputs import HOLMES_CATEGORIES, build_balanced_holmes_manifests


class TestVideoSeekLaveCPU(unittest.TestCase):
    def test_parse_tool_call_variants(self):
        xml_call = (
            "<tool_call><function=focus>"
            "<parameter=query>What text is on the phone screen?</parameter>"
            "<parameter=start_time>42.5</parameter>"
            "<parameter=end_time>58.0</parameter>"
            "</function></tool_call>"
        )
        p1 = parse_tool_call(xml_call)
        self.assertEqual(p1["name"], "focus")
        self.assertAlmostEqual(p1["arguments"]["start_time"], 42.5)
        self.assertAlmostEqual(p1["arguments"]["end_time"], 58.0)

        json_call = '<tool_call>{"name": "skim", "arguments": {"query": "find car", "start_time": 10, "end_time": 90}}</tool_call>'
        p2 = parse_tool_call(json_call)
        self.assertEqual(p2["name"], "skim")
        self.assertEqual(p2["arguments"]["end_time"], 90)

        ans_call = "<tool_call><function=answer></function></tool_call>"
        self.assertEqual(parse_tool_call(ans_call)["name"], "answer")

    def test_extract_choice_letter(self):
        valid = ("A", "B", "C", "D", "E", "F")
        self.assertEqual(extract_choice_letter("F", valid), "F")
        self.assertEqual(extract_choice_letter("The final answer is (C).", valid), "C")
        self.assertEqual(extract_choice_letter("Option B: colleague", valid), "B")
        self.assertIsNone(extract_choice_letter("No letter here", valid))

    def test_split_budget_preserves_exact_frame_count(self):
        # Peak outside planner window -> 8 + 8 = 16 exact frames
        plan = plan_split_budget_timestamps(
            "focus", 100.0, 116.0, 200.0, 16, [45.0, 160.0]
        )
        self.assertTrue(plan["rerouted"])
        self.assertEqual(len(plan["timestamps"]), 16)
        self.assertEqual(len(plan["latent_windows"]), 2)
        self.assertEqual(plan["timestamps"], sorted(plan["timestamps"]))

        # Peak already inside planner window -> no reroute, still 16 exact frames
        plan_in = plan_split_budget_timestamps(
            "focus", 100.0, 116.0, 200.0, 16, [108.0]
        )
        self.assertFalse(plan_in["rerouted"])
        self.assertEqual(len(plan_in["timestamps"]), 16)

    def test_entropy_and_peak_nms(self):
        ts = [10.0, 15.0, 50.0, 90.0]
        lifts = [2.0, 1.8, 1.5, -0.2]
        peaks = select_latent_peaks(ts, lifts, min_sep_s=12.0, max_peaks=2)
        self.assertEqual(peaks, [0, 2])
        self.assertLess(normalized_positive_entropy([5.0, 0.0, 0.0, 0.0]), 0.01)
        self.assertGreater(normalized_positive_entropy([1.0, 1.0, 1.0, 1.0]), 0.99)

    def test_select_top_bank_prefers_focus_on_close_timestamps(self):
        bank = [
            dict(timestamp_s=10.0, tool="overview"),
            dict(timestamp_s=10.5, tool="focus"),
            dict(timestamp_s=50.0, tool="skim"),
        ]
        picked = select_top_bank_entries(bank, [1.5, 1.2, 2.0], max_frames=2, dedup_tol_s=1.5)
        self.assertEqual(picked, [1, 2])  # focus at 10.5 replaces overview at 10.0

    def test_concat_mixed_resolution_embeds(self):
        # Image 1: grid (1, 16, 28) -> (16/2)*(28/2)=112 tokens; Image 2: grid (1, 24, 42) -> 12*21=252 tokens
        s1 = torch.randn(112, 64)
        s2 = torch.randn(252, 64)
        emb, g = concat_multi_image_embeds([s1, s2], [(1, 16, 28), (1, 24, 42)])
        self.assertEqual(tuple(emb.shape), (364, 64))
        self.assertEqual(g.tolist(), [[1, 16, 28], [1, 24, 42]])

    def test_balanced_blind_holmes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            vdir = tdp / "videos"
            vdir.mkdir()
            rows = []
            qid = 0
            for v in range(30):
                vid = f"vid_{v:02d}"
                (vdir / f"{vid}.mp4").write_bytes(b"")
                for cat in HOLMES_CATEGORIES:
                    qid += 1
                    rows.append(
                        {
                            "video ID": vid,
                            "Question ID": qid,
                            "Question Type": cat,
                            "Question": f"Q {qid}?",
                            "Options": {"A": "a", "B": "b", "C": "c", "D": "d"},
                            "Answer": "A",
                            "Explanation": "secret",
                        }
                    )
            jpath = tdp / "test.json"
            jpath.write_text(json.dumps(rows))
            pilot, eval_m = build_balanced_holmes_manifests(
                jpath, vdir, pilot_per_cat=2, eval_per_cat=10, probe_duration=False
            )
            self.assertEqual(pilot["n"], 14)
            self.assertEqual(eval_m["n"], 70)
            self.assertEqual(pilot["labels"], 0)
            self.assertEqual(eval_m["labels"], 0)
            self.assertEqual(set(pilot["order"]) & set(eval_m["order"]), set())
            self.assertNotIn("Answer", json.dumps(pilot))
            self.assertNotIn("Explanation", json.dumps(eval_m))


if __name__ == "__main__":
    unittest.main()
