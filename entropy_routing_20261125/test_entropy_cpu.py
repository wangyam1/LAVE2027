"""Local stdlib CPU unit tests for entropy_routing_20261125."""

import json
from types import SimpleNamespace
import unittest

from missing_fact_routing_20261124.target import target_from_question

from entropy_routing_20261125 import experiment, multi_action, policy, report


def _distance_sorted(idx, ordered):
    from bisect import bisect_left
    p = bisect_left(ordered, idx)
    return min(abs(ordered[i] - idx) for i in (p - 1, p) if 0 <= i < len(ordered))


def _make_sample(source):
    def sample(lo, hi):
        lo_i, hi_i = int(round(lo * 10)), int(round(hi * 10))
        if hi_i - lo_i < 32:
            return sorted(set(range(lo_i, hi_i + 1)))
        step = 31
        return [lo_i + (i * (hi_i - lo_i) + step // 2) // step for i in range(32)]
    return sample


class EntropyRoutingCPUTest(unittest.TestCase):
    def test_calibrated_lift_nominates_negative_margin_and_enforces_controls(self):
        # Zero baseline strongly biased toward N (-4.0), real peak has raw margin -0.5 (lift +3.5)
        zero = (-4.0, 0.0)
        real = [(-3.5, 0.0) for _ in range(32)]
        real[5] = (-0.5, 0.0)   # lift +3.5 (peak)
        real[12] = (-1.0, 0.0)  # lift +3.0
        real[24] = (-1.2, 0.0)  # lift +2.8
        eligible = list(range(31))
        nom = policy.nominate(real, zero, eligible)
        self.assertIsNotNone(nom)
        self.assertEqual(nom.slot, 5)
        self.assertAlmostEqual(nom.gain, 3.5)
        self.assertAlmostEqual(nom.margin, -0.5)

        # Passes when beating shifted, scrambled, wrong_target
        dec = policy.approve(nom, real, zero, (-2.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), eligible)
        self.assertTrue(dec.accepted)
        self.assertIn(len(dec.addresses), (1, 2, 4))
        self.assertEqual(dec.per_gap_frames * len(dec.addresses), 16)
        # NMS: all chosen addresses are pairwise non-adjacent
        for idx, a in enumerate(dec.addresses):
            for b in dec.addresses[idx + 1:]:
                self.assertGreater(abs(a - b), 1)

        # Rejected by scrambled
        dec_scram = policy.approve(nom, real, zero, (-0.5, 0.0), (-2.0, 0.0), (-4.0, 0.0), eligible)
        self.assertFalse(dec_scram.accepted)
        self.assertEqual(dec_scram.addresses, ())

        # Rejected by wrong_target
        dec_wrong = policy.approve(nom, real, zero, (-2.0, 0.0), (-0.4, 0.0), (-4.0, 0.0), eligible)
        self.assertFalse(dec_wrong.accepted)

        # Rejected by shifted
        real_sh = list(real)
        real_sh[(5 + 16) % 32] = (-0.5, 0.0)
        nom_sh = policy.nominate(real_sh, zero, eligible)
        dec_sh = policy.approve(nom_sh, real_sh, zero, (-2.0, 0.0), (-2.0, 0.0), (-4.0, 0.0), eligible)
        self.assertFalse(dec_sh.accepted)

    def test_entropy_tertiles_select_k_1_2_4(self):
        eligible = list(range(20))
        zero = (-3.0, 0.0)
        # Sharp peak -> low entropy -> K=1
        sharp = [(-3.0, 0.0) for _ in range(32)]
        sharp[4] = (6.0, 0.0)
        nom = policy.nominate(sharp, zero, eligible)
        dec = policy.approve(nom, sharp, zero, (-2.0, 0.0), (-2.0, 0.0), zero, eligible)
        self.assertTrue(dec.accepted)
        self.assertLess(dec.entropy_norm, 1.0 / 3.0)
        self.assertEqual(len(dec.addresses), 1)
        self.assertEqual(dec.per_gap_frames, 16)

        # Flat/diffuse lifts -> high entropy -> K=4
        diffuse = [(-1.0, 0.0) for _ in range(32)]
        diffuse[3] = (-0.5, 0.0)
        diffuse[19] = (-2.0, 0.0)  # shifted slot lower
        nom_d = policy.nominate(diffuse, zero, eligible)
        dec_d = policy.approve(nom_d, diffuse, zero, (-2.0, 0.0), (-2.0, 0.0), zero, eligible)
        self.assertTrue(dec_d.accepted)
        self.assertGreaterEqual(dec_d.entropy_norm, 2.0 / 3.0)
        self.assertEqual(len(dec_d.addresses), 4)
        self.assertEqual(dec_d.per_gap_frames, 4)

    def test_multiclip_action_contract_and_factorial(self):
        source = [i * 100 for i in range(32)]
        sample = _make_sample(source)
        geometry = dict(source=source, fps=10.0, count=3200)
        text = dict(query="Count the green onions picked up.", start_time=0.0, end_time=310.0)
        split = SimpleNamespace(distance_sorted=_distance_sorted)

        for k in (1, 2, 4):
            def choose(eligible, k=k):
                addrs = policy._pick_nms_gaps(eligible, k)
                return dict(addresses=addrs, per_gap_frames=16 // k)

            action, sampler, audit = experiment.plan_actions(split, text, geometry, sample, choose)
            for arm in multi_action.ARMS:
                self.assertEqual(len(action["indices"][arm]), 32)
                self.assertEqual(len(set(action["indices"][arm])), 32)
            self.assertEqual(len(action["arguments"]["state_split"]["clips"]), 1 + k)
            self.assertEqual(len(action["arguments"]["clock_split"]["clips"]), 1 + k)
            multi_action.bind(action, sampler, text)

    def test_multiframe_report_parser(self):
        times = [float(i) for i in range(32)]
        # Valid 5-frame citation (previously rejected!)
        raw5 = json.dumps({"frames": [3, 4, 5, 6, 7], "observation": "The buyer picked up three green onions."})
        st, obs = report.serialize_focus_report(raw5, times)
        self.assertEqual(st, "observed")
        self.assertIn("Frame 3 (3.000s)", obs)
        self.assertIn("Frame 7 (7.000s)", obs)
        # Valid 2-sentence observation
        raw2s = json.dumps({"frames": [1, 2, 8], "observation": "First fact visible.\nSecond fact visible."})
        st2, obs2 = report.serialize_focus_report(raw2s, times)
        self.assertEqual(st2, "observed")
        self.assertEqual(len(obs2.splitlines()), 1)
        # Reject >8 frames
        raw9 = json.dumps({"frames": list(range(9)), "observation": "Too many frames."})
        self.assertEqual(report.serialize_focus_report(raw9, times)[0], "invalid")
        # Reject floats / timestamps
        raw_ts = json.dumps({"frames": [183.4], "observation": "Timestamp."})
        self.assertEqual(report.serialize_focus_report(raw_ts, times)[0], "invalid")
        # Reject option label
        raw_opt = json.dumps({"frames": [1], "observation": "The answer is Option A."})
        self.assertEqual(report.serialize_focus_report(raw_opt, times)[0], "invalid")
        # UNRESOLVED
        self.assertEqual(report.serialize_focus_report("UNRESOLVED", times), ("unresolved", report.UNRESOLVED_REPORT))


if __name__ == "__main__":
    unittest.main()
