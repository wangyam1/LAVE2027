"""Exercise the paired LAVE case with native contracts and synthetic observations."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from . import experiment
from .report import UNRESOLVED_REPORT
from .target import source_check_messages, target_from_question


ROOT = Path(__file__).resolve().parents[1]
SPLIT, = ROOT.glob("receiver_training/**/split_budget_v1/triad_case.py")
NATIVE, = ROOT.glob("receiver_training/**/native_itt_content_utility_v1/run_itt_utility.py")
sys.path[:0] = [str(p) for p in (
    SPLIT.parent, SPLIT.parent.parents[2], NATIVE.parent, ROOT / "adjacent_source_routing_20261124"
)]
import native_focus_io as io
import run_itt_utility as native
import split_action as split
import triad_case as triad
from focus_contract import action_contract
from route import gaps


CASE = dict(id="cpu-case", video_uid="synthetic", question="What is the price of the blue pen?",
            choices=["choice one", "choice two"], duration_annotated=324,
            video_path="synthetic.mp4", video_bytes=0, split="development")
TEXT = dict(query="Read the price caption for the blue pen.", start_time=0, end_time=8)
REPORTS = ["A pen appears on the desk.", "There is a close-up of its cap."]
FPS = 4
GEOMETRY = dict(source=list(range(0, 1280, 40)), fps=FPS, count=1296)
TARGET = target_from_question(CASE["question"])
WRONG_TARGET = target_from_question("What is the price of the apple?")


def sample(lo, hi):
    left, right, size = native.focus_indices(lo, hi, FPS, GEOMETRY["count"])
    if size == 1:
        return [left]
    return [left + i * (right - left - 1) // (size - 1) for i in range(size)]


def uncalled(*_):
    raise AssertionError("This stage must not run")


class FactorialTest(unittest.TestCase):
    def selection(self, eligible, blocked_by=None):
        real = [(0.0, 1.0)] * 32
        real[10] = (2.0, 0.0)
        if blocked_by == "shift":
            real[26] = real[10]
        controls = dict(scrambled=(0.0, 1.0), wrong=(0.0, 1.0), wrong_zero=(0.0, 1.0))
        if blocked_by in ("scrambled", "wrong"):
            controls[blocked_by] = real[10]
        rows = [dict(image_grid_thw=[[1, 2, 2]] * 32, text=REPORTS[0]), dict(text=REPORTS[1])]
        d = SimpleNamespace(s=SimpleNamespace(rope=lambda _: object()))
        vlm = SimpleNamespace(model=SimpleNamespace(config=object()), processor=object())
        row = SimpleNamespace(requests=[{}])
        with (
            patch.object(experiment.receiver, "render", side_effect=[({"label": "real"}, ()), ({"label": "wrong"}, ())]) as render,
            patch.object(experiment.receiver, "probe_states", return_value=dict(real=real, zero=(0.0, 1.0))),
            patch.object(experiment.receiver, "probe_controls", return_value=controls) as control,
        ):
            audit = experiment.select_address(d, row, vlm, None, TARGET, WRONG_TARGET, TEXT, rows, eligible)
        self.assertEqual(render.call_count, 2)
        self.assertEqual(render.call_args_list[0].args[-4:-1], (TARGET, TEXT["query"], REPORTS))
        self.assertEqual(render.call_args_list[1].args[-4:-1], (WRONG_TARGET, TEXT["query"], REPORTS))
        self.assertEqual(control.call_args.args[4], 10)
        return audit

    def plan(self, text=TEXT, *, choose=None, source_status=None):
        return experiment.plan_actions(
            split, triad, action_contract, gaps, text, GEOMETRY, sample,
            choose if choose is not None else self.selection, source_status=source_status,
        )

    def run_case(self, plan, text=TEXT, *, ordinal=0, mismatch=False):
        action, sampler, _ = plan
        calls, qa = [], []

        def observe(mode, arm, indices, query):
            self.assertEqual(query, text)
            calls.append((mode, arm))
            jpeg = [sha256(f"jpeg{frame}".encode()).hexdigest() for frame in indices]
            rgb = [sha256(f"rgb{frame}".encode()).hexdigest() for frame in indices]
            if mismatch and mode == "targeted" and arm == "clock_split":
                rgb[0] = "changed"
            observation = {
                ("targeted", "text"): "UNRESOLVED",
                ("targeted", "state_split"): '{"frames":[16],"observation":"The price label reads 30."}',
                ("targeted", "clock_split"): '{"frames":[32],"observation":"Out of bounds."}',
            }.get((mode, arm), UNRESOLVED_REPORT)
            return dict(status="focus_generated", calls=1, indices=indices,
                        clocks=[f"{frame / FPS:.1f}s" for frame in indices], jpeg_sha256=jpeg,
                        rgb_sha256=rgb, observation=observation, capped=False, record={"mock": True})

        def answer(mode, arm, messages):
            qa.append((mode, arm))
            self.assertTrue(all(isinstance(message["content"], str) for message in messages))
            user = messages[1]["content"]
            self.assertIn("Question:\n" + CASE["question"], user)
            self.assertIn("Observation 1:\n" + REPORTS[0], user)
            self.assertIn("Observation 2:\n" + REPORTS[1], user)
            if mode == "targeted" and arm == "state_split":
                self.assertIn("Observation 3:\nFocus evidence from Frame 16", user)
            return "A"

        args = (triad, io, native, CASE, REPORTS, TARGET, FPS, ordinal, action, sampler, text, observe, answer)
        if mismatch:
            with self.assertRaisesRegex(ValueError, "different visual evidence"):
                experiment.run_factorial(*args)
            self.assertEqual(qa, [])
            return None, calls, qa
        return experiment.run_factorial(*args), calls, qa

    def test_exact_split_and_cited_fact(self):
        plan = self.plan()
        action, sampler, audit = plan
        self.assertTrue(audit["decision"]["accepted"])
        self.assertEqual(audit["address"], 9)
        frames, keys = triad.bind(action, sampler, TEXT)
        anchor = split.evenly_native(frames["text"])
        self.assertEqual(len(set(keys.values())), 3)
        self.assertTrue(all(len(values) == 32 and set(anchor) <= set(values) for values in frames.values()))
        self.assertTrue(all(row["query"] == TEXT["query"] for row in action["arguments"].values()))
        result, calls, qa = self.run_case(plan, ordinal=6)
        self.assertEqual(len(calls), 6)
        self.assertEqual([mode for mode, _ in calls], ["targeted"] * 3 + ["native"] * 3)
        self.assertEqual(result["cells"]["text"]["targeted"]["status"], "unresolved")
        self.assertEqual(result["cells"]["clock_split"]["targeted"]["status"], "invalid")
        self.assertEqual(result["cells"]["state_split"]["targeted"]["observation"],
                         f"Focus evidence from Frame 16 ({frames['state_split'][16] / FPS:.3f}s). The price label reads 30.")
        self.assertEqual(len(qa), result["unique_qa_calls"])
        self.assertEqual(len(qa), 2)
        self.assertEqual(result["cells"]["text"]["native"]["qa_reused_from"], dict(mode="targeted", arm="text"))

    def test_controls_reject_to_the_same_native_action(self):
        for control in ("shift", "scrambled", "wrong"):
            with self.subTest(control=control):
                plan = self.plan(choose=lambda eligible: self.selection(eligible, control))
                action, _, audit = plan
                self.assertFalse(audit["decision"]["accepted"])
                self.assertIsNone(audit["address"])
                self.assertEqual(action["indices"]["text"], action["indices"]["state_split"])
                self.assertEqual(action["indices"]["text"], action["indices"]["clock_split"])
                result, calls, qa = self.run_case(plan)
                self.assertEqual(sum(run["physical_calls"] for run in result["focus"].values()), 2)
                self.assertEqual(len(calls), 2)
                self.assertEqual(len(qa), 1)

    def test_already_resolved_short_and_invalid_actions(self):
        cases = ((TEXT, "RESOLVED", 2), (dict(TEXT, end_time=2), None, 2), (None, None, 0))
        for text, source_status, calls in cases:
            with self.subTest(text=text, source_status=source_status):
                plan = self.plan(text, choose=uncalled, source_status=source_status)
                self.assertIsNone(plan[2])
                physical, _ = triad.bind(plan[0], plan[1], text)
                self.assertEqual(len({tuple(v) for v in physical.values()}), int(bool(calls)))
                _, focus, qa = self.run_case(plan, text)
                self.assertEqual(len(focus), calls)
                self.assertEqual(len(qa), int(bool(calls)))
        self.assertTrue(experiment.needs_verification("RESOLVED because..."))

    def test_visual_receipt_mismatch_stops_before_qa(self):
        self.run_case(self.plan(), mismatch=True)

    def test_targets_and_prompt_boundary(self):
        path = ROOT / "adjacent_source_routing_20261124" / "focus.case_review.private.json"
        cases = json.loads(path.read_text())["cases"]
        price = target_from_question(cases["dev-16"]["question"])
        count = target_from_question(cases["dev-04"]["question"])
        self.assertEqual(price.attribute, "price")
        self.assertIn("'Jinhao 80' fountain pen", price.entity)
        self.assertEqual(count.attribute, "count")
        self.assertIn("pick up with one hand", count.entity)
        self.assertEqual(count.time_condition, "when the three buyers went to the vegetable and fruit stall")
        self.assertIn(count.question, source_check_messages(count, REPORTS)[1]["content"])
        last = dict(type="text", text="Video Subtitles:\n\n\nQuestion:\n" + TEXT["query"]
                    + experiment.NATIVE_REPORT_INSTRUCTION + "Begin every line with a timestamp.")
        rendered = dict(indices=[0], clocks=["0.0s"], jpeg_sha256=["jpeg"], rgb_sha256=["rgb"],
                        messages=[dict(role="user", content=[dict(type="image"), last])])
        snapshot = deepcopy(rendered)
        revised = experiment.targeted_view(rendered, price, FPS)
        self.assertEqual(rendered, snapshot)
        self.assertEqual(revised["messages"][0]["content"][:-1], rendered["messages"][0]["content"][:-1])
        request = revised["messages"][0]["content"][-1]["text"]
        self.assertTrue(request.startswith("Video Subtitles:\n\n\nQuestion:\n" + TEXT["query"] + "\n\n"))
        self.assertNotIn("Begin every line with a timestamp.", request)
        self.assertIn("Frame 0 (0.000s)", request)
        self.assertIn(price.question, request)


if __name__ == "__main__":
    unittest.main()
