"""Count feasible corrective time actions on the frozen exposed cases."""

from collections import Counter
import json
import os

from .focus_side_policy import plan_focus_sides
from .live_run import need
from .refocus_run import load, seal


def build_menu(resources):
    """Compute the geometry before trial outcomes, with no answer-key access."""
    result = {}
    for row, _, prior, candidates in resources[-1]:
        seen = {i for indices in prior["routes"]["indices"].values() for i in indices}
        plans = {}
        for origin, (view, _) in candidates.items():
            plans[origin] = plan_focus_sides(
                cited_indices=view["cited_indices"], source_indices=row.media["overview_indices"],
                focus_indices=view["frame_indices"], fps=row.media["fps"], frame_count=row.media["frame_count"],
                other_focus_indices=seen,
            )
        result[row.cid] = plans
    return result


def main():
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU geometry requires hidden CUDA")
    resources = load()
    result = build_menu(resources)
    plans = [p for plans in result.values() for p in plans.values()]
    counts = Counter(p["reason"] or "planned" for p in plans)
    need(not resources[1].s.torch.cuda.is_initialized(), "Geometry unexpectedly initialized CUDA")
    print(json.dumps(dict(status="focus_side_cpu_geometry", videos=len(result), claims=len(plans), counts=counts,
                          roster_sha256=seal(result), labels=0, holdout=0, gpu=0), sort_keys=True))


if __name__ == "__main__":
    main()
