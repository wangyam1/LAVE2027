"""Capture time-bound slots during a normal Focus call for a later Planner."""

from contextlib import nullcontext, redirect_stdout
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch

from .native_runtime import _receipt


BOUND_FIELDS = (
    "messages", "rendered", "images", "input_ids_sha256", "position_ids_sha256", "mm_token_type_ids_sha256",
    "pixel_values_sha256", "pixel_values_shape", "image_grid_thw", "maximum_new_tokens", "tools",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def transfer_images(source, dest, images):
    """Copy exact first-Focus JPEGs into a fresh native worker's asset directory."""
    (dest / "images").mkdir(mode=0o700)
    for image in images:
        digest = image["sha256"]
        require(len(digest) == 64 and image["path"] == f"images/{digest}.jpg", "Unexpected cached image reference")
        original, copied = source / image["path"], dest / image["path"]
        require(not original.is_symlink() and not original.parent.is_symlink(), "Unexpected link to a cached image")
        data = original.read_bytes()
        require(sha256(data).hexdigest() == digest, "Changed first-Focus image")
        if not copied.exists():
            with copied.open("xb") as stream:
                stream.write(data)
        require(sha256(copied.read_bytes()).hexdigest() == digest, "Changed copied first-Focus JPEG")


def capture_first_focus(
    candidate, *, work, native, runtime, historical, model, capture, occupancy, sink=None,
):
    """Repeat a sealed development input once and capture states in that ordinary generation.

    The caller controls execution and media scope. The recorded first Focus
    output remains the shared history and is eligible only when this new
    generation reproduces it exactly.
    """
    source = Path(candidate["directory"])
    reference = candidate["focus"]
    origin = candidate["origin"]
    require(source.is_absolute() and reference["path"] == f"outputs/focus_{origin}.json",
            "Unexpected first-Focus call")
    raw = (source / reference["path"]).read_bytes()
    require(sha256(raw).hexdigest() == reference["sha256"], "Changed first-Focus receipt")
    archived = json.loads(raw)
    cap = runtime.COMMON["source_cap"]
    require(archived["audit_passed"] and archived["actual_model_generation"] and archived["visual_calls"] == 1
            and len(candidate["frame_indices"]) == len(candidate["images"]) == len(archived["images"])
            and candidate["images"] == archived["images"] and archived["maximum_new_tokens"] == cap,
            "Unexpected first-Focus visual input")
    directory = Path(work)
    directory.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    worker = runtime.attach(native, historical, directory, model, "focus", 1)
    transfer_images(source, directory, archived["images"])
    one = []
    original_generate = model.model.generate

    def tap(**kwargs):
        require(not one, "A single Focus generation may capture only one state sequence")
        with capture(model.model, kwargs) as observed:
            result = original_generate(**kwargs)
        one.append(observed)
        return result

    occupancy()
    with (nullcontext() if sink is None else redirect_stdout(sink)):
        with patch.object(model.model, "generate", side_effect=tap) as invocation:
            _, ref = worker.generate("first_focus", "focus", archived["messages"], cap, archived["images"])
    occupancy()
    live = _receipt(directory, ref, "focus", "first_focus", archived["messages"], cap)
    require(all(live[k] == archived[k] for k in BOUND_FIELDS)
            and invocation.call_count == len(one) == 1, "First-Focus prefill no longer matches the sealed request")
    captured, = one
    count = len(candidate["frame_indices"])
    require(captured["verified"] and captured["vision_calls"] == 1 and len(captured["states"]) == count,
            "Could not map one captured state to each actual frame")
    matched = live["generated_token_ids"] == archived["generated_token_ids"]
    audit = dict(record=ref, frozen_record=reference, matched_generated_tokens=matched,
                 visual_prefills=captured["vision_calls"], states=count, frame_indices=candidate["frame_indices"],
                 image_tokens=captured["image_tokens"])
    return captured["states"] if matched else None, audit
