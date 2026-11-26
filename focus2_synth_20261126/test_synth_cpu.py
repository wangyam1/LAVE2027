"""Local CPU unit tests for focus2_synth_20261126."""

import unittest

from focus2_synth_20261126.synth_run import (
    PRIORITY_NOTE,
    combine_synth_observation,
    direct_visual_qa_messages,
    synth_focus_messages,
)


class Focus2SynthCPUTest(unittest.TestCase):
    def test_synth_focus_messages_option_blind(self):
        msgs = synth_focus_messages(["10.0s", "11.0s"], "How many pink masks were given away?")
        self.assertEqual(len(msgs), 1)
        body = msgs[0]["content"]
        self.assertEqual(len(body), 1 + 2 * 2 + 1)
        self.assertIn("How many pink masks were given away?", body[-1]["text"])
        self.assertIn("Direct visual answer:", body[-1]["text"])
        self.assertNotIn("Options:", body[-1]["text"])

    def test_direct_visual_qa_messages_has_options(self):
        case = dict(question="How many?", choices=["1", "2", "3"])
        msgs = direct_visual_qa_messages(["10.0s"], case, ["r1", "r2"], "f1")
        txt = msgs[0]["content"][-1]["text"]
        self.assertIn("A. 1", txt)
        self.assertIn("C. 3", txt)

    def test_combine_synth_observation(self):
        self.assertEqual(combine_synth_observation("old", "", False), "old")
        self.assertEqual(combine_synth_observation("old", "new", True), "old")
        self.assertIn("Focused dense inspection:\nnew", combine_synth_observation("old", "new", False))
        self.assertEqual(
            combine_synth_observation("No relevant content found.", "new", False),
            "Focused dense inspection:\nnew",
        )
        self.assertIn("Observation 3", PRIORITY_NOTE)


if __name__ == "__main__":
    unittest.main()
