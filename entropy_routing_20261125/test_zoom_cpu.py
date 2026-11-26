"""Local CPU unit tests for Focus-2 dense verification."""

import unittest

from entropy_routing_20261125 import zoom_verify_run as zr


class ZoomVerifyCPUTest(unittest.TestCase):
    def test_parse_timestamps_and_densest_cluster(self):
        obs = "12.0s: No relevant content found.\n284.2s: pen intro\n303.8s: price range\n9999.0s: out of bounds"
        ts = zr.parse_focus1_timestamps(obs, 1800.0)
        self.assertEqual(ts, [284.2, 303.8])
        lo, hi = zr.densest_cluster_window(ts, 1800.0)
        self.assertAlmostEqual(lo, 274.2)
        self.assertAlmostEqual(hi, 313.8)

        # Single timestamp expands to at least 20s
        lo1, hi1 = zr.densest_cluster_window([362.0], 1000.0)
        self.assertAlmostEqual(lo1, 352.0)
        self.assertAlmostEqual(hi1, 372.0)

        # Dense cluster preferred over isolated outlier
        lo2, hi2 = zr.densest_cluster_window([10.0, 300.0, 305.0, 312.0, 900.0], 1000.0)
        self.assertAlmostEqual(lo2, 290.0)
        self.assertAlmostEqual(hi2, 322.0)

    def test_pick_unseen_frames_and_combine(self):
        frames = zr.pick_unseen_frames(10.0, 30.0, fps=10.0, frame_count=1000, seen={100, 150, 200})
        self.assertEqual(len(frames), 16)
        self.assertEqual(len(set(frames)), 16)
        self.assertTrue(all(f not in {100, 150, 200} for f in frames))
        self.assertEqual(zr.combine_observation("284.2s: pen", "No relevant content found.", False), "284.2s: pen")
        self.assertEqual(zr.combine_observation("284.2s: pen", "Price is 29 yuan.", True), "284.2s: pen")
        self.assertEqual(zr.combine_observation("No relevant content found.", "Price is 29 yuan.", False), "Price is 29 yuan.")
        self.assertIn("Focused dense inspection:", zr.combine_observation("284.2s: pen", "Price is 29 yuan.", False))


if __name__ == "__main__":
    unittest.main()
