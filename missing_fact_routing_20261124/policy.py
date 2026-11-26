"""Choose a physical time gap based on missing-fact Y/N scores."""

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Nomination:
    address: int
    slot: int
    gain: float
    shifted: float


@dataclass(frozen=True)
class Decision:
    nomination: Nomination
    scrambled: float
    wrong_question: float
    accepted: bool


def _log_odds(pair):
    if len(pair) != 2 or any(type(v) not in (int, float) or not isfinite(v) for v in pair):
        raise ValueError("Expected finite Y and N logits")
    log_odds = pair[0] - pair[1]
    if not isfinite(log_odds):
        raise ValueError("Invalid Y/N margin")
    return log_odds


def nominate(real, zero, eligible):
    """Use the strongest positively supported endpoint and an earlier-gap tie-break."""
    eligible = tuple(eligible)
    if len(real) != 32 or any(type(i) is not int or not 0 <= i < 31 for i in eligible):
        raise ValueError("The Source must provide 32 real slots and valid adjacent gap addresses")
    if not eligible:
        return None
    blank = _log_odds(zero)
    margin = [_log_odds(x) for x in real]
    candidates = (
        (margin[slot] - blank, -address, -slot)
        for address in eligible
        for slot in (address, address + 1)
        if margin[slot] > 0
    )
    best = max(candidates, default=None)
    if best is None or best[0] <= 0:
        return None
    gain, address, slot = best[0], -best[1], -best[2]
    return Nomination(address, slot, gain, margin[(slot + 16) % 32] - blank)


def approve(nomination, zero, scrambled, wrong, wrong_zero):
    """Require a strict win over each matched content or address control."""
    scrambled_gain = _log_odds(scrambled) - _log_odds(zero)
    wrong_gain = _log_odds(wrong) - _log_odds(wrong_zero)
    accepted = nomination.gain > max(0, nomination.shifted, scrambled_gain, wrong_gain)
    return Decision(nomination, scrambled_gain, wrong_gain, accepted)
