"""Adapt the native LAVE runtime to multi-frame targeted Focus and identical-visual checks."""

from missing_fact_routing_20261124.native_runtime import NativeAdapter as _BaseAdapter, QA_TEMPLATE, _receipt, _require

from .experiment import NATIVE_REPORT_INSTRUCTION, targeted_view


class NativeAdapter(_BaseAdapter):
    """Use the updated multi-frame targeted prompt while keeping all native receipts identical."""

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


__all__ = ["NativeAdapter", "QA_TEMPLATE", "_receipt"]
