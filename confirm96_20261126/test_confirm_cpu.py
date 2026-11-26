"""Local stdlib CPU unit tests for confirm96_20261126."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest

from confirm96_20261126 import confirm_audit as ca, confirm_inputs as ci, confirm_score as cs


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class Confirm96CPUTest(unittest.TestCase):
    def test_abi_case_preserves_fields(self):
        c = dict(
            id="cgb_confirm_001", rank=1, seed_index=0, qid="8139", video_uid="vidA",
            question="How many apples are on the table?", choices=["1", "2", "3"],
            duration_annotated=700.0, video_path="/tmp/vidA.mp4", video_bytes=100,
            video_sha256="a" * 64, split="confirmation",
        )
        abi = ci.abi_case(c)
        self.assertEqual(abi["split"], "development")
        self.assertEqual(c["split"], "confirmation")
        self.assertEqual(set(abi), {"id", "video_uid", "question", "choices",
                                    "duration_annotated", "video_path", "video_bytes", "split"})

    def test_confirm_foils_disjoint_video_and_question(self):
        cases = [
            dict(id=f"cgb_confirm_{i+1:03d}", video_uid=f"v{i}",
                 question=f"How many items appear in clip {i}?", split="confirmation")
            for i in range(96)
        ]
        plan = ci.plan_confirm_foils(cases, FakeTokenizer())
        self.assertEqual(len(plan), 96)
        for c in cases:
            f = plan[c["id"]]
            self.assertNotEqual(f["foil_id"], c["id"])
            self.assertNotEqual(f["question"], c["question"])

    def test_score_summary_math(self):
        row1 = {}
        row2 = {}
        for ep in cs.ENDPOINTS:
            row1[("state_split", ep)] = True
            row1[("text", ep)] = False
            row1[("clock_split", ep)] = False
            row2[("state_split", ep)] = True
            row2[("text", ep)] = True
            row2[("clock_split", ep)] = False
        items = [
            (row1, {"state_split": (1,), "text": (2,), "clock_split": (3,)}),
            (row2, {"state_split": (1,), "text": (2,), "clock_split": (3,)}),
        ]
        out = cs._summarize_qa(items)
        self.assertEqual(out["correct"]["state_split"]["focus12_equal_budget"], 2)
        self.assertEqual(out["routing"]["focus12_equal_budget"]["state_split_vs_text"]["win"], 1)
        self.assertEqual(out["routing"]["focus12_equal_budget"]["state_split_vs_clock_split"]["win"], 2)

    def test_verify_source_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "outputs").mkdir()
            images = []
            for i in range(32):
                data = f"fake-jpeg-{i}".encode()
                h = sha256(data).hexdigest()
                rel = f"images/{h}.jpg"
                (root / rel).write_bytes(data)
                images.append(dict(path=rel, sha256=h, size=[64, 64]))
            msgs = [{"role": "user", "content": [{"type": "text", "text": "overview"}]}]
            rec = dict(
                key="source_shared", stage="source", audit_passed=True, audit_pending=False,
                actual_model_generation=True, visual_input=True, visual_calls=1, latent_present=False,
                tools=None, messages=msgs, maximum_new_tokens=2048, text="report",
                input_ids=[1, 2, 3], input_tokens=3, generated_token_ids=[4, 5], output_tokens=2,
                generation_cap=False, images=images,
            )
            raw = json.dumps(rec).encode()
            (root / "outputs" / "source_shared.json").write_bytes(raw)
            ref = dict(path="outputs/source_shared.json", sha256=sha256(raw).hexdigest())
            verified = ca._verify_source_receipt(root, ref, "source_shared", msgs, 2048)
            self.assertEqual(verified["text"], "report")
            # Tamper one JPEG and ensure verification fails
            (root / images[0]["path"]).write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                ca._verify_source_receipt(root, ref, "source_shared", msgs, 2048)


if __name__ == "__main__":
    unittest.main()
