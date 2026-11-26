"""Pure-CPU unit tests for lvb_vllm_20261126 (category balance, interleaving, prompts)."""

from collections import Counter
import unittest

from lvb_vllm_20261126.lvb_inputs import select_category_balanced_cases
from lvb_vllm_20261126.lvb_run import distance_sorted, focus_messages


class LVBVllmCPUTest(unittest.TestCase):
    def test_category_balanced_150_and_shard_parity(self):
        cats = [f"C{i:02d}" for i in range(17)]
        rows = []
        rid = 0
        for ci, c in enumerate(cats):
            # Vary available unique videos (6..15) and rows (14..30)
            n_vids = 6 + (ci % 10)
            for j in range(18):
                rows.append(dict(
                    id=f"q_{rid}",
                    video_path=f"v_{ci}_{j % n_vids}.mp4",
                    duration=800.0 if j < 8 else 300.0,
                    question_category=c,
                ))
                rid += 1
        sel = select_category_balanced_cases(rows, n_total=150, min_duration=180.0)
        self.assertEqual(len(sel), 150)
        counts = Counter(r["question_category"] for r in sel)
        self.assertEqual(len(counts), 17)
        self.assertEqual(min(counts.values()), 8)
        self.assertEqual(max(counts.values()), 9)
        # Check even/odd 2-GPU shards are also balanced across all 17 categories
        s0 = Counter(r["question_category"] for r in sel[0::2])
        s1 = Counter(r["question_category"] for r in sel[1::2])
        self.assertEqual(len(s0), 17)
        self.assertEqual(len(s1), 17)
        self.assertLessEqual(max(abs(s0[c] - s1[c]) for c in cats), 1)

    def test_focus_messages_format(self):
        msgs = focus_messages([0, 24], 24.0, "What color is the car?")
        body = msgs[0]["content"]
        self.assertEqual(len(body), 1 + 2 * 2 + 1)
        self.assertEqual(body[1]["text"], "0.0s")
        self.assertEqual(body[3]["text"], "1.0s")
        self.assertIn("Question:\nWhat color is the car?\n\n", body[-1]["text"])

    def test_distance_sorted(self):
        from entropy_routing_20261125 import multi_action, policy

        self.assertEqual(distance_sorted(5, [0, 10]), 5)
        self.assertEqual(distance_sorted(2, [0, 10]), 2)
        menu = {i: list(range(100 + i * 20, 100 + i * 20 + 16)) for i in range(6)}
        picked = policy.select_clock_gaps(menu, 2, {100, 110}, multi_action.evenly_native, distance_sorted)
        self.assertEqual(len(picked), 2)
        self.assertGreater(abs(picked[0] - picked[1]), 1)


if __name__ == "__main__":
    unittest.main()
