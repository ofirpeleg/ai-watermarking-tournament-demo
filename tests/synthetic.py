"""
Synthetic-distribution corpus generator used by the test suite.

None of the verification tests need the real model: what has to be verified
is the watermarking LOGIC (gates, tournament, hashing, detection stats), and
that logic is pure Python living in watermark_core.decide_step. Feeding it
hand-built probability distributions is faster, deterministic, and exercises
the exact same code path generate.py uses.
"""

from __future__ import annotations

import math
import random

import watermark_core as wc

VOCAB_SIZE = 200


def shannon_entropy_bits(probs: list[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def synth_step_distribution(rng: random.Random, top_k: int, confident: bool):
    """One synthetic step's top-k (token_id, prob) list, sorted descending.

    confident=True mimics a step where the model is nearly certain (mirrors
    the real ' Name' (p=0.995) vs ' Names' (p=0.0018) example from the
    project brief) -- low entropy, should hit the entropy gate.

    confident=False mimics a step with several live options -- higher
    entropy, several candidates within min_ratio of the leader, should
    survive both gates and run a tournament.
    """
    ids = rng.sample(range(VOCAB_SIZE), top_k)
    if confident:
        leader = rng.uniform(0.985, 0.999)
        rest_mass = 1.0 - leader
        tail = [rng.random() for _ in range(top_k - 1)]
        s = sum(tail) or 1.0
        tail = sorted((t / s * rest_mass for t in tail), reverse=True)
        probs = [leader] + tail
    else:
        raw = [rng.random() ** 0.6 for _ in range(top_k)]
        s = sum(raw)
        probs = sorted((r / s for r in raw), reverse=True)
    return list(zip(ids, probs))


def make_watermarked_corpus(
    n_steps: int,
    key: str,
    rounds: int,
    window: int,
    entropy_min: float,
    min_ratio: float,
    top_k: int = 8,
    confident_fraction: float = 0.55,
    seed: int = 0,
):
    """Run decide_step over n_steps synthetic distributions. Returns
    (token_ids, steps)."""
    rng = random.Random(seed)
    generated: list[int] = []
    steps: list[wc.StepResult] = []
    for _ in range(n_steps):
        confident = rng.random() < confident_fraction
        topk = synth_step_distribution(rng, top_k, confident)
        entropy = shannon_entropy_bits([p for _, p in topk])
        step = wc.decide_step(
            topk=topk,
            entropy=entropy,
            generated_so_far=generated,
            key=key,
            rounds=rounds,
            window=window,
            entropy_min=entropy_min,
            min_ratio=min_ratio,
        )
        steps.append(step)
        generated.append(step.winner_id)
    return generated, steps


def make_random_corpus(n_steps: int, seed: int = 0) -> list[int]:
    """Plain random token stream -- stands in for unwatermarked ("normal")
    text. Not run through decide_step at all."""
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_SIZE) for _ in range(n_steps)]
