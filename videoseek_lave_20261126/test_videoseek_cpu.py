"""CPU unit tests for multi-step VideoSeek + LAVE (no GPU required)."""

import json
from pathlib import Path
import tempfile
import unittest

import torch

from videoseek_lave_20261126.agent import (
    choose_geometric_clock_peak,
    extract_choice_letter,
    normalized_positive_entropy,
    parse_tool_call,
    plan_split_budget_timestamps,
    select_latent_peaks,
    select_top_bank_entries,
    strip_think_tags,
)
from videoseek_lave_20261126.baselines_holmes import _thin_image_history
from videoseek_lave_20261126.prompts import filter_nonempty_observations
from videoseek_lave_20261126.engine_ext import concat_multi_image_embeds
from videoseek_lave_20261126.holmes_inputs import HOLMES_CATEGORIES, build_balanced_holmes_manifests
from videoseek_lave_20261126.mme_inputs import parse_mme_options


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

        # Qwen2.5-VL positional & natural-language tool call formats
        p3 = parse_tool_call('skim("man in black beanie and hoodie", 0.0, 136.0)')
        self.assertEqual(p3["name"], "skim")
        self.assertAlmostEqual(p3["arguments"]["start_time"], 0.0)
        self.assertAlmostEqual(p3["arguments"]["end_time"], 136.0)
        self.assertEqual(p3["arguments"]["query"], "man in black beanie and hoodie")

        p4 = parse_tool_call('focus(213.6, 213.6, "finding a key from her bag")')
        self.assertEqual(p4["name"], "focus")
        self.assertAlmostEqual(p4["arguments"]["start_time"], 213.6)
        self.assertEqual(p4["arguments"]["query"], "finding a key from her bag")

        p5 = parse_tool_call("focus(40.0, 128.0)")
        self.assertEqual(p5["name"], "focus")
        self.assertAlmostEqual(p5["arguments"]["end_time"], 128.0)

        p6 = parse_tool_call('skim with the query "The vehicle broke down" from 0.0s to 243.0s.')
        self.assertEqual(p6["name"], "skim")
        self.assertAlmostEqual(p6["arguments"]["end_time"], 243.0)
        self.assertEqual(p6["arguments"]["query"], "The vehicle broke down")

        p7 = parse_tool_call('skim with query="supernatural" and start_time=82.0s and end_time=120.0s')
        self.assertEqual(p7["name"], "skim")
        self.assertAlmostEqual(p7["arguments"]["start_time"], 82.0)

        self.assertEqual(parse_tool_call("answer")["name"], "answer")
        self.assertEqual(parse_tool_call("answer with start_time=0.0s and end_time=248.0s.")["name"], "answer")

        # Qwen2.5-VL <tool_call>{...} without closing </tool_call> tag
        p8 = parse_tool_call('<tool_call>{"name": "skim", "arguments": {"query": "core theme", "start_time": 0.0, "end_time": 136.0}}')
        self.assertEqual(p8["name"], "skim")
        self.assertAlmostEqual(p8["arguments"]["end_time"], 136.0)
        self.assertEqual(p8["arguments"]["query"], "core theme")

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

    def test_parse_mme_options(self):
        opts = parse_mme_options(["A. Apples.", "B. Candles.", "C. Berries.", "D. All same."])
        self.assertEqual(opts, {"A": "Apples.", "B": "Candles.", "C": "Berries.", "D": "All same."})

    def test_strip_think_tags(self):
        raw = "<think>Let me analyze step by step...</think>\nFinal answer: (D)"
        self.assertEqual(strip_think_tags(raw), "Final answer: (D)")
        self.assertEqual(extract_choice_letter(raw, ("A", "B", "C", "D")), "D")
        self.assertEqual(strip_think_tags("<think>unclosed reasoning only"), "unclosed reasoning only")

    def test_choose_geometric_clock_peak(self):
        # Video 0..200s, planner inspects 10..30s -> largest unvisited gap is 30..200s -> midpoint 115.0s
        peaks = choose_geometric_clock_peak(200.0, 10.0, 30.0, visited_windows=[])
        self.assertEqual(peaks, [115.0])
        # With 90..180s also visited -> largest remaining gap is 30..90s (60s) -> midpoint 60.0s
        peaks2 = choose_geometric_clock_peak(200.0, 10.0, 30.0, visited_windows=[(90.0, 180.0)])
        self.assertEqual(peaks2, [60.0])

    def test_thin_image_history_respects_budget(self):
        entries = [
            dict(tool="overview", note="0-100", images=list(range(32)), timestamps=list(range(32))),
            dict(tool="skim", note="10-50", images=list(range(16)), timestamps=list(range(16))),
            dict(tool="focus", note="20-35", images=list(range(16)), timestamps=list(range(16))),
            dict(tool="focus", note="60-75", images=list(range(16)), timestamps=list(range(16))),
        ]
        kept, pruned = _thin_image_history(entries, max_images=64)
        total_kept = sum(len(e["images"]) for e in kept)
        self.assertLessEqual(total_kept, 64)
        self.assertGreater(pruned, 0)
        # Most recent focus entry stays completely intact
        self.assertEqual(len(kept[-1]["images"]), 16)

    def test_filter_nonempty_observations(self):
        blocks = [
            "Observation from `overview()`:\n0.0s: A man walks in.",
            "Observation from `skim` (0.0s-50.0s):\nNo relevant content found.",
            "Observation from `focus` (10.0s-20.0s):\nThe man holds a blue mug.",
            "Observation from `focus` (60.0s-70.0s):\n  no relevant content found  ",
        ]
        out = filter_nonempty_observations(blocks)
        self.assertIn("A man walks in.", out)
        self.assertIn("blue mug", out)
        self.assertNotIn("No relevant content found", out)
        self.assertNotIn("no relevant content found", out)


if __name__ == "__main__":
    unittest.main()
