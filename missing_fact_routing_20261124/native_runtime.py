"""Adapt an already admitted LAVE runtime to the matched Focus and QA experiment."""

from contextlib import nullcontext, redirect_stdout
from hashlib import sha256
import json
from math import prod
from pathlib import Path
import re

from .experiment import MODES, NATIVE_REPORT_INSTRUCTION, targeted_view
from .target import source_check_messages


SHA256 = re.compile(r"[0-9a-f]{64}")
QA_TEMPLATE = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return isinstance(value, str) and SHA256.fullmatch(value) is not None


def _receipt(directory, ref, stage, key, messages, cap):
    relative = f"outputs/{key}.json"
    _require(ref.get("path") == relative and _digest(ref.get("sha256")), "Wrong physical record reference")
    path = directory / relative
    _require(not path.is_symlink() and not path.parent.is_symlink(), "Unexpected linked generation receipt")
    data = path.read_bytes()
    _require(sha256(data).hexdigest() == ref["sha256"], "Native generation receipt changed on disk")
    record = json.loads(data)
    visual = stage == "focus"
    _require(
        record["key"] == key and record["stage"] == stage
        and record["audit_passed"] is True and record["audit_pending"] is False
        and record["actual_model_generation"] is True and record["visual_input"] is visual
        and record["visual_calls"] == int(visual) and record["latent_present"] is False
        and record["tools"] is None and record["messages"] == messages and record["maximum_new_tokens"] == cap
        and isinstance(record["text"], str) and len(record["input_ids"]) == record["input_tokens"] > 0
        and 0 < len(record["generated_token_ids"]) == record["output_tokens"] <= cap
        and record["generation_cap"] is (record["output_tokens"] == cap),
        "Native generation receipt does not match the submitted request",
    )
    return record


class NativeAdapter:
    """Execute isolated native calls with identical visual input across reporting modes."""

    def __init__(self, *, native, io, helper, reference, joint_render, joint_bridge, runtime,
                 historical, work, model, reader, fps, count, case, reports, target, occupancy, sink=None):
        self.native, self.io, self.helper = native, io, helper
        self.reference = Path(reference)
        self.joint_render, self.joint_bridge, self.runtime = joint_render, joint_bridge, runtime
        self.historical, self.work, self.model = historical, Path(work), model
        self.reader, self.fps, self.count = reader, fps, count
        self.case, self.reports, self.target = case, tuple(reports), target
        self.occupancy, self.sink = occupancy, sink
        self.focus_ledger = {mode: {} for mode in MODES}
        self._visual_budget = None
        self._visual_signatures = {}

    def _worker(self, stage, mode, arm):
        _require(mode in MODES and arm in self.joint_bridge.ARMS, "Unknown Focus condition")
        directory = self.work / stage / mode / arm
        directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        worker = self.runtime.attach(self.native, self.historical, directory, self.model, stage, 1)
        return directory, worker

    def _stdout(self):
        return nullcontext() if self.sink is None else redirect_stdout(self.sink)

    def _prompt(self, rendered, mode, query):
        original = rendered["messages"][0]
        request, separator, _ = original["content"][-1]["text"].rpartition(NATIVE_REPORT_INSTRUCTION)
        _require(separator and request.endswith("Question:\n" + query["query"]), "The native Focus query changed")
        if mode == "native":
            return rendered
        directed = targeted_view(rendered, self.target, self.fps)
        message = directed["messages"][0]
        _require(
            all(directed[field] == rendered[field] for field in ("indices", "clocks", "jpeg_sha256", "rgb_sha256"))
            and message["role"] == original["role"] and message["content"][:-1] == original["content"][:-1]
            and message["content"][-1]["text"].startswith(request + "\n\n"),
            "Targeted instructions changed the original query or visual evidence",
        )
        return directed

    def observe(self, mode, arm, indices, query):
        directory, worker = self._worker("focus", mode, arm)
        rendered = self.joint_render.render(self.native, self.io, self.helper, self.reference, directory,
                                            query, indices, self.reader, self.fps, self.count)
        prompt = self._prompt(rendered, mode, query)
        self.occupancy()
        with self._stdout():
            observation = self.joint_bridge.one_focus(worker, arm, prompt)
        self.occupancy()
        ref = observation["record"]
        record = _receipt(directory, ref, "focus", f"focus_{arm}", prompt["messages"],
                          self.runtime.COMMON["source_cap"])
        grids, pixels = record["image_grid_thw"], record["pixel_values_shape"]
        mm_types = record["mm_token_type_ids"]
        _require(
            len(grids) == len(record["images"]) == len(indices)
            and all(len(grid) == 3 and all(type(v) is int and v > 0 for v in grid) for grid in grids)
            and len(pixels) == 2 and all(type(v) is int and v > 0 for v in pixels)
            and _digest(record["pixel_values_sha256"])
            and len(mm_types) == record["input_tokens"] and all(v in (0, 1) for v in mm_types)
            and [image["sha256"] for image in record["images"]] == prompt["jpeg_sha256"]
            and record["text"] == observation["observation"] and record["generation_cap"] == observation["capped"],
            "Persisted Focus evidence differs from the sealed visual request",
        )
        patches, visual_tokens = sum(map(prod, grids)), sum(mm_types)
        _require(patches == pixels[0] and 0 < visual_tokens < record["input_tokens"], "Invalid Focus tensor budget")
        budget = len(indices), patches, visual_tokens
        if self._visual_budget is None:
            self._visual_budget = budget
        _require(budget == self._visual_budget, "Focus visual budget changed across conditions")
        signature = (prompt["clocks"], prompt["jpeg_sha256"], prompt["rgb_sha256"],
                     grids, pixels, record["pixel_values_sha256"])
        expected = self._visual_signatures.setdefault(tuple(indices), signature)
        _require(signature == expected, "Identical Focus indices produced different visual evidence")
        self.focus_ledger[mode][arm] = dict(
            frames=len(indices), patches=patches, visual_tokens=visual_tokens, input_tokens=record["input_tokens"],
            nonvisual_input_tokens=record["input_tokens"] - visual_tokens,
            output_tokens=record["output_tokens"], image_grid_thw=grids, pixel_values_shape=pixels,
            pixel_values_sha256=record["pixel_values_sha256"], record=ref,
        )
        return observation

    def _text_generation(self, directory, worker, key, messages):
        cap = self.runtime.COMMON["qa_cap"]
        self.occupancy()
        with self._stdout():
            _, ref = worker.generate(key, "qa", messages, cap)
        self.occupancy()
        record = _receipt(directory, ref, "qa", key, messages, cap)
        template = self.model.tokenizer.apply_chat_template(messages, **QA_TEMPLATE)
        ids = self.model.tokenizer.encode(template, add_special_tokens=False)
        _require(
            worker.state["model_calls"] == worker.state["qa_calls"] == len(worker.records) == 1
            and not record["images"] and record["rendered"] == template and record["input_ids"] == ids,
            "Did not execute the exact native one-call text inference",
        )
        return dict(text=record["text"], capped=record["generation_cap"], record=ref,
                    input_tokens=record["input_tokens"], output_tokens=record["output_tokens"])

    def check_source(self, messages):
        _require(messages == source_check_messages(self.target, self.reports), "Changed the Source verification request")
        self.work.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory = self.work / "source_check"
        worker = self.runtime.attach(self.native, self.historical, directory, self.model, "qa", 1)
        return self._text_generation(directory, worker, "source_check", messages)

    def answer(self, mode, arm, messages):
        blank = self.io.qa_messages(self.native, self.case, self.reports, "")
        _require(
            len(messages) == len(blank) == 2 and messages[0] == blank[0]
            and isinstance(messages[1]["content"], str) and messages[1]["content"].startswith(blank[1]["content"]),
            "QA changed the question, options, Source reports or native system instruction",
        )
        observation = messages[1]["content"][len(blank[1]["content"]):]
        _require(messages == self.io.qa_messages(self.native, self.case, self.reports, observation),
                 "QA must differ only in its third text observation")
        directory, worker = self._worker("qa", mode, arm)
        outcome = self._text_generation(directory, worker, f"qa_{arm}", messages)
        choice = None if outcome["capped"] else self.native.option(outcome["text"], self.case["choices"])
        return dict(status="generated", answer=choice, capped=outcome["capped"], record=outcome["record"],
                    input_tokens=outcome["input_tokens"], output_tokens=outcome["output_tokens"])
