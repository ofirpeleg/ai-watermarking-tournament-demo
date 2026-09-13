"""
Shared watermarking core.

This module contains every piece of logic that generate.py and detect.py must
agree on bit-for-bit: the keyed hash, the context-window rule, the entropy /
viability gates, the tournament, and the detection statistics.

Deliberately has ZERO dependency on torch or transformers. generate.py needs
the model to produce logits; detect.py must never load a model at all. Putting
the shared logic here (instead of duplicating it in both files) is what makes
that separation safe -- there is only one implementation of "how a token gets
scored", so generation and detection cannot silently drift apart the way they
did in the bug this project is built to demonstrate (see README, "Context
must be generated-only").
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence


# ---------------------------------------------------------------------------
# The keyed hash and the context-window rule
# ---------------------------------------------------------------------------

def keyed_bit(key: str, round_num: int, context: Sequence[int], token_id: int) -> int:
    """The one low-level primitive the whole scheme is built on.

    g_bit(round, context, token) = low bit of SHA256(key | round | context | token_id)

    Deterministic and stateless: same inputs always produce the same bit, on
    either side (generation or detection), with no model involved.
    """
    h = hashlib.sha256()
    h.update(key.encode("utf-8"))
    h.update(b"|")
    h.update(str(round_num).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(str(t) for t in context).encode("utf-8"))
    h.update(b"|")
    h.update(str(token_id).encode("utf-8"))
    return h.digest()[0] & 1


def context_window(generated_tokens: Sequence[int], window: int) -> tuple:
    """The last `window` GENERATED tokens -- never the prompt.

    Called identically from generate.py (with the tokens produced so far in
    this run) and detect.py (with the tokens seen so far in the document
    under analysis). A detector only ever receives the output text, never the
    prompt behind it -- so if generation let the prompt leak into this
    context, detection could never reproduce it. Bounding the window (default
    4) also means a single edited token only disturbs the next ~window
    tokens' worth of hashing, not everything downstream of it.
    """
    if window <= 0:
        return ()
    if not generated_tokens:
        return ()
    return tuple(generated_tokens[-window:])


def bit_sum(key: str, rounds: int, context: Sequence[int], token_id: int) -> int:
    """Sum of the `rounds` keyed bits for one (context, token) pair."""
    return sum(keyed_bit(key, r, context, token_id) for r in range(rounds))


def g_score(key: str, rounds: int, context: Sequence[int], token_id: int) -> float:
    """Fraction of the `rounds` keyed bits that are 1, for one (context, token) pair.

    In [0, 1]. Under the null (a token that was not selected for its bits)
    this is unbiased with expectation 0.5. This is the per-token statistic
    detection aggregates.
    """
    return bit_sum(key, rounds, context, token_id) / rounds


# ---------------------------------------------------------------------------
# Gating + tournament (generation side, but pure / model-free so it is
# directly unit-testable with synthetic probability distributions)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    token_id: int
    prob: float
    bit_sum: Optional[int] = None  # filled in only for candidates that entered the tournament


@dataclass
class Match:
    a_token_id: int
    a_bit_sum: int
    b_token_id: int
    b_bit_sum: int
    winner_token_id: int
    reason: str  # "higher bit-sum" | "tie on bit-sum, higher probability"


@dataclass
class StepResult:
    entropy: float
    gate: Optional[str]          # None if watermarked, else "entropy" | "viability"
    watermarked: bool
    winner_id: int
    rounds: int
    all_topk: list               # list[Candidate] as presented (pre-filter), for reporting
    viable: list                 # list[Candidate] that passed the ratio gate (bit_sum set)
    excluded: list                # list[Candidate] that failed the ratio gate
    matches: list                # list[Match], empty unless watermarked
    winner_g: Optional[float] = None  # bit_sum/rounds of the winner, only if watermarked


def run_tournament(viable: Sequence[Candidate]) -> tuple[Candidate, list]:
    """Single-elimination bracket over candidates, seeded by probability (desc).

    Each match: higher bit-sum wins; a tie on bit-sum is broken by higher
    model probability. Because "higher bit-sum" is a transitive comparison,
    the true maximum survives to the final round regardless of bracket
    seeding -- so the champion is exactly the global bit-sum winner (with the
    global tie-break), while still producing a real match-by-match log.
    """
    round_participants = sorted(viable, key=lambda c: -c.prob)
    matches: list[Match] = []
    while len(round_participants) > 1:
        next_round = []
        for i in range(0, len(round_participants), 2):
            if i + 1 >= len(round_participants):
                next_round.append(round_participants[i])  # odd one out gets a bye
                continue
            a, b = round_participants[i], round_participants[i + 1]
            if a.bit_sum > b.bit_sum:
                winner, reason = a, "higher bit-sum"
            elif b.bit_sum > a.bit_sum:
                winner, reason = b, "higher bit-sum"
            else:
                winner = a if a.prob >= b.prob else b
                reason = "tie on bit-sum, higher probability"
            matches.append(Match(a.token_id, a.bit_sum, b.token_id, b.bit_sum, winner.token_id, reason))
            next_round.append(winner)
        round_participants = next_round
    return round_participants[0], matches


def decide_step(
    topk: Sequence[tuple[int, float]],
    entropy: float,
    generated_so_far: Sequence[int],
    key: str,
    rounds: int,
    window: int,
    entropy_min: float,
    min_ratio: float,
) -> StepResult:
    """Apply gate 1 (entropy), gate 2 (viability), then the tournament.

    `topk` is [(token_id, prob), ...] already sorted by prob descending
    (the caller -- generate.py, or a test -- is responsible for producing
    this from a real or synthetic distribution). `generated_so_far` must
    contain ONLY previously generated tokens, never the prompt.

    Pure function: no model, no I/O. This is what makes the alignment test
    possible without downloading a model.
    """
    all_topk = [Candidate(tid, p) for tid, p in topk]

    if entropy < entropy_min:
        leader = all_topk[0]
        return StepResult(entropy, "entropy", False, leader.token_id, rounds, all_topk, [], all_topk, [])

    leader_p = all_topk[0].prob
    viable_ids = [c for c in all_topk if c.prob >= min_ratio * leader_p]
    excluded = [c for c in all_topk if c not in viable_ids]

    if len(viable_ids) < 2:
        leader = all_topk[0]
        return StepResult(entropy, "viability", False, leader.token_id, rounds, all_topk, viable_ids, excluded, [])

    ctx = context_window(generated_so_far, window)
    for c in viable_ids:
        c.bit_sum = bit_sum(key, rounds, ctx, c.token_id)

    champion, matches = run_tournament(viable_ids)
    return StepResult(
        entropy, None, True, champion.token_id, rounds, all_topk, viable_ids, excluded, matches,
        winner_g=champion.bit_sum / rounds,
    )


# ---------------------------------------------------------------------------
# Detection statistics
# ---------------------------------------------------------------------------

def sequence_to_pairs(token_ids: Sequence[int], window: int) -> list[tuple[tuple, int]]:
    """Turn a flat token sequence into the (context, token) pairs detection scores.

    Position i's context is the `window` tokens immediately before it in this
    same sequence -- exactly mirroring what generate.py used at generation
    time, since by the time a document is being detected, "generated so far"
    and "document so far" are the same thing.
    """
    pairs = []
    for i, tid in enumerate(token_ids):
        ctx = context_window(token_ids[:i], window)
        pairs.append((ctx, tid))
    return pairs


def deduplicate_pairs(pairs: Sequence[tuple[tuple, int]]) -> list[tuple[tuple, int]]:
    """Keep only the first occurrence of each distinct (context, token) pair.

    Required for a valid z-test: the null model treats each scored pair as an
    independent fair-coin draw. A model looping on one token repeats the same
    (context, token) pair over and over, which is not independent evidence --
    without dedup it reads as an overwhelming, entirely spurious detection.
    """
    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


@dataclass
class DetectionResult:
    tokens_analyzed: int
    rounds: int
    g1_count: int
    g0_count: int
    mean_g: float
    expected_mean: float
    sigma: float
    z: float
    p_value: float
    p_threshold: float
    verdict: bool
    deduped_from: Optional[int] = None  # original pair count, if dedup removed any


def compute_detection(
    pairs: Sequence[tuple[tuple, int]],
    key: str,
    rounds: int,
    p_threshold: float,
    dedup: bool = True,
) -> DetectionResult:
    from scipy.stats import norm

    original_n = len(pairs)
    if dedup:
        pairs = deduplicate_pairs(pairs)
    n = len(pairs)
    if n == 0:
        raise ValueError("No tokens to analyze.")

    g1 = 0
    g0 = 0
    for context, token_id in pairs:
        ones = bit_sum(key, rounds, context, token_id)
        g1 += ones
        g0 += rounds - ones

    total_bits = g1 + g0
    mean_g = g1 / total_bits
    expected_mean = 0.5
    sigma = 0.5 / math.sqrt(total_bits)
    z = (mean_g - expected_mean) / sigma
    # norm.sf (survival function, 1 - cdf computed directly) keeps precision
    # into the far tail; naive `1 - norm.cdf(z)` underflows to exactly 0.0
    # for z beyond ~9 due to catastrophic cancellation, even when the true
    # p-value (e.g. 1e-50) is still representable.
    p_value = float(norm.sf(z))  # one-sided: is Mean G significantly ABOVE 0.5?

    return DetectionResult(
        tokens_analyzed=n,
        rounds=rounds,
        g1_count=g1,
        g0_count=g0,
        mean_g=mean_g,
        expected_mean=expected_mean,
        sigma=sigma,
        z=z,
        p_value=p_value,
        p_threshold=p_threshold,
        verdict=bool(p_value < p_threshold),
        deduped_from=original_n if (dedup and original_n != n) else None,
    )
