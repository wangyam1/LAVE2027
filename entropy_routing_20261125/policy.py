"""Calibrated-lift nomination, content controls, and entropy-guided multi-gap allocation."""

from dataclasses import dataclass
from math import exp, isfinite, log


ALLOWED_K = (1, 2, 4)
EXTRA_BUDGET = 16


@dataclass(frozen=True)
class Nomination:
    address: int
    slot: int
    gain: float
    shifted: float
    margin: float


@dataclass(frozen=True)
class Decision:
    nomination: Nomination
    scrambled: float
    wrong_question: float
    accepted: bool
    entropy_norm: float
    addresses: tuple[int, ...]
    per_gap_frames: int


def _log_odds(pair):
    if len(pair) != 2 or any(type(v) not in (int, float) or not isfinite(v) for v in pair):
        raise ValueError("Expected finite Y and N logits")
    value = pair[0] - pair[1]
    if not isfinite(value):
        raise ValueError("Invalid Y/N margin")
    return value


def _slot_lifts(real, zero):
    if len(real) != 32:
        raise ValueError("The Source must provide 32 real visual slots")
    blank = _log_odds(zero)
    margins = [_log_odds(pair) for pair in real]
    lifts = [m - blank for m in margins]
    return margins, lifts


def _gap_lifts(lifts, eligible):
    eligible = tuple(eligible)
    if any(type(i) is not int or not 0 <= i < 31 for i in eligible) or len(set(eligible)) != len(eligible):
        raise ValueError("Expected distinct adjacent gap addresses in 0..30")
    return {a: max(lifts[a], lifts[a + 1]) for a in eligible}


def nominate(real, zero, eligible):
    """Nominate the strongest positive-lift eligible gap without an uncalibrated raw-margin cutoff."""
    margins, lifts = _slot_lifts(real, zero)
    eligible = tuple(eligible)
    if not eligible:
        return None
    _gap_lifts(lifts, eligible)
    candidates = (
        (lifts[slot], -address, -slot)
        for address in eligible
        for slot in (address, address + 1)
    )
    best = max(candidates, default=None)
    if best is None or best[0] <= 0.0:
        return None
    gain, address, slot = best[0], -best[1], -best[2]
    return Nomination(address, slot, gain, lifts[(slot + 16) % 32], margins[slot])


def _normalized_entropy(values):
    n = len(values)
    if n <= 1:
        return 0.0
    mx = max(values)
    weights = [exp(v - mx) for v in values]
    total = sum(weights)
    probs = [w / total for w in weights]
    entropy = -sum(p * log(p) for p in probs if p > 0.0)
    return max(0.0, min(1.0, entropy / log(n)))


def _pick_nms_gaps(ranked, k):
    """Select up to k non-adjacent gaps (|a - b| > 1), falling back only if menu is small."""
    chosen = []
    for a in ranked:
        if all(abs(a - b) > 1 for b in chosen):
            chosen.append(a)
            if len(chosen) == k:
                return chosen
    for a in ranked:
        if a not in chosen:
            chosen.append(a)
            if len(chosen) == k:
                return chosen
    return chosen


def approve(nomination, real, zero, scrambled, wrong, wrong_zero, eligible):
    """Require strict wins over shifted, scrambled, and wrong-target controls, then allocate K gaps by entropy."""
    _, lifts = _slot_lifts(real, zero)
    gap_scores = _gap_lifts(lifts, eligible)
    if nomination.address not in gap_scores:
        raise ValueError("Nominated address is outside the eligible gap menu")
    scrambled_gain = _log_odds(scrambled) - _log_odds(zero)
    wrong_gain = _log_odds(wrong) - _log_odds(wrong_zero)
    accepted = nomination.gain > max(0.0, nomination.shifted, scrambled_gain, wrong_gain)
    ordered = sorted(gap_scores)
    entropy_norm = _normalized_entropy([gap_scores[a] for a in ordered])
    if not accepted:
        return Decision(nomination, scrambled_gain, wrong_gain, False, entropy_norm, (), 0)

    # Textbook normalized Shannon entropy tertiles control concentration vs coverage (K in {1, 2, 4}).
    raw_k = 1 if entropy_norm < (1.0 / 3.0) else (2 if entropy_norm < (2.0 / 3.0) else 4)
    ranked = sorted(ordered, key=lambda a: (gap_scores[a], -a), reverse=True)
    positive_ranked = [a for a in ranked if gap_scores[a] > 0.0]
    nms_pool = _pick_nms_gaps(positive_ranked, raw_k)
    k = max(cand for cand in ALLOWED_K if cand <= len(nms_pool))
    chosen = tuple(nms_pool[:k])
    return Decision(nomination, scrambled_gain, wrong_gain, True, entropy_norm, chosen, EXTRA_BUDGET // k)


def select_clock_gaps(menu, k, seen_init, evenly_native, distance_sorted):
    """Matched greedy farthest-point novelty clock with exact same K, per-gap frames, and NMS rule."""
    if k not in ALLOWED_K or len(menu) < k:
        raise ValueError("Invalid clock gap count for this menu")
    per_gap = EXTRA_BUDGET // k
    seen = sorted(seen_init)
    chosen = []
    keys = sorted(menu)
    for _ in range(k):
        pool_keys = [a for a in keys if a not in chosen and all(abs(a - b) > 1 for b in chosen)]
        if not pool_keys:
            pool_keys = [a for a in keys if a not in chosen]

        def novelty(addr):
            sub = evenly_native(menu[addr], per_gap)
            dists = sorted(distance_sorted(frame, seen) for frame in sub)
            return dists[len(dists) // 2], sum(dists), -addr

        best = max(pool_keys, key=novelty)
        chosen.append(best)
        seen = sorted(set(seen) | set(evenly_native(menu[best], per_gap)))
    return tuple(chosen)
