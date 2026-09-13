"""
Verification suite. Everything here runs against synthetic probability
distributions (see synthetic.py) -- no model, no network, no GPU -- because
what needs verifying is the watermarking logic itself (gates, tournament,
hashing, detection statistics), and that logic is pure Python in
watermark_core.py. generate.py just wires it up to real model logits.
"""

import math

import pytest

import watermark_core as wc
from synthetic import make_random_corpus, make_watermarked_corpus, shannon_entropy_bits

KEY = "my-demo-secret-key"
ROUNDS = 256
WINDOW = 4
ENTROPY_MIN = 0.5
MIN_RATIO = 0.1
P_THRESHOLD = 0.01
N_STEPS = 700


# ---------------------------------------------------------------------------
# 1. Alignment -- the test that would have caught the prompt-context bug
# ---------------------------------------------------------------------------

def test_alignment_detector_recovers_exact_generation_time_g():
    """decide_step must hash using ONLY generated_so_far. If it leaked a
    prompt prefix into the context (the historical bug), the g the detector
    recomputes from the bare output sequence would not match the g the
    tournament actually used to pick the winner."""
    tokens, steps = make_watermarked_corpus(
        N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=1
    )
    pairs = wc.sequence_to_pairs(tokens, WINDOW)

    watermarked_steps = [s for s in steps if s.watermarked]
    assert len(watermarked_steps) > 50, "test setup should produce plenty of tournament steps"

    for i, (s, (ctx, tid)) in enumerate(zip(steps, pairs)):
        assert tid == s.winner_id
        if not s.watermarked:
            continue
        recomputed_g = wc.g_score(KEY, ROUNDS, ctx, tid)
        assert recomputed_g == pytest.approx(s.winner_g), (
            f"step {i}: generation-time g ({s.winner_g}) != detector-recomputed g "
            f"({recomputed_g}) -- context must be generated-tokens-only on both sides"
        )


def test_context_window_excludes_everything_before_it():
    # A window of 2 over [10, 20, 30, 40] at position 4 must be exactly (30, 40),
    # never anything from further back (i.e. never a "prompt").
    assert wc.context_window([10, 20, 30, 40], 2) == (30, 40)
    assert wc.context_window([10, 20, 30, 40], 0) == ()
    assert wc.context_window([], 4) == ()
    assert wc.context_window([10, 20], 4) == (10, 20)


# ---------------------------------------------------------------------------
# 2. Separation
# ---------------------------------------------------------------------------

def test_separation_watermarked_detects_normal_does_not():
    wm_tokens, _ = make_watermarked_corpus(N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=1)
    normal_tokens = make_random_corpus(N_STEPS, seed=2)

    wm_result = wc.compute_detection(wc.sequence_to_pairs(wm_tokens, WINDOW), KEY, ROUNDS, P_THRESHOLD)
    normal_result = wc.compute_detection(wc.sequence_to_pairs(normal_tokens, WINDOW), KEY, ROUNDS, P_THRESHOLD)

    assert wm_result.verdict is True
    assert wm_result.p_value < P_THRESHOLD
    assert normal_result.verdict is False
    assert normal_result.p_value >= P_THRESHOLD


# ---------------------------------------------------------------------------
# 3. Key dependence
# ---------------------------------------------------------------------------

def test_key_dependence():
    wm_tokens, _ = make_watermarked_corpus(N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=1)

    right_key = wc.compute_detection(wc.sequence_to_pairs(wm_tokens, WINDOW), KEY, ROUNDS, P_THRESHOLD)
    wrong_key = wc.compute_detection(wc.sequence_to_pairs(wm_tokens, WINDOW), "not-the-key", ROUNDS, P_THRESHOLD)

    assert right_key.verdict is True
    assert wrong_key.verdict is False


# ---------------------------------------------------------------------------
# 4. Window dependence
# ---------------------------------------------------------------------------

def test_window_dependence():
    wm_tokens, _ = make_watermarked_corpus(N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=1)

    right_window = wc.compute_detection(wc.sequence_to_pairs(wm_tokens, WINDOW), KEY, ROUNDS, P_THRESHOLD)
    wrong_window = wc.compute_detection(wc.sequence_to_pairs(wm_tokens, WINDOW + 3), KEY, ROUNDS, P_THRESHOLD)

    assert right_window.verdict is True
    assert wrong_window.verdict is False


# ---------------------------------------------------------------------------
# 5. Repetition / dedup
# ---------------------------------------------------------------------------

def _find_biased_repeating_token(key, rounds, window, min_bias=0.55, search_size=500):
    """Find a token id whose steady-state self-repeating (context, token) pair
    happens to hash above 0.5, so the no-dedup false-positive demo is
    deterministic rather than a coin flip."""
    for tid in range(search_size):
        steady_ctx = tuple([tid] * window)
        g = wc.g_score(key, rounds, steady_ctx, tid)
        if g >= min_bias:
            return tid
    raise RuntimeError("no suitable token found in search range")


def test_repetition_false_positive_without_dedup_only():
    repeat_token = _find_biased_repeating_token(KEY, ROUNDS, WINDOW)
    tokens = [repeat_token] * 300

    pairs = wc.sequence_to_pairs(tokens, WINDOW)

    with_dedup = wc.compute_detection(pairs, KEY, ROUNDS, P_THRESHOLD, dedup=True)
    without_dedup = wc.compute_detection(pairs, KEY, ROUNDS, P_THRESHOLD, dedup=False)

    # Deduplicated: this "text" is really only a handful of distinct
    # (context, token) observations -- nowhere near enough evidence.
    assert with_dedup.tokens_analyzed <= WINDOW + 1
    assert with_dedup.verdict is False

    # Undeduplicated: the same handful of observations counted 300 times
    # each masquerades as 300 independent ones -- a confident false positive.
    assert without_dedup.tokens_analyzed == 300
    assert without_dedup.verdict is True
    assert without_dedup.p_value < 1e-10


# ---------------------------------------------------------------------------
# 6. Quality -- the viability gate must hold in practice, not just in theory
# ---------------------------------------------------------------------------

def test_quality_no_low_probability_token_ever_wins():
    _, steps = make_watermarked_corpus(N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=3)

    for s in steps:
        leader_p = s.all_topk[0].prob
        winner_p = next(c.prob for c in s.all_topk if c.token_id == s.winner_id)
        # every emitted token, watermarked or not, must be within min_ratio of
        # the leader -- this is exactly what stops a 0.995 vs 0.0018 step
        # (the ' Name' / ' Names' example) from swapping in the unlikely one
        assert winner_p >= MIN_RATIO * leader_p - 1e-9


def test_entropy_gate_keeps_confident_steps_unwatermarked():
    # A near-certain distribution (entropy well under the default 0.5 bit
    # threshold) must always keep the model's own top token.
    topk = [(1, 0.995), (2, 0.003), (3, 0.001), (4, 0.001)]
    h = shannon_entropy_bits([p for _, p in topk])
    assert h < ENTROPY_MIN
    step = wc.decide_step(topk, h, [], KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO)
    assert step.watermarked is False
    assert step.gate == "entropy"
    assert step.winner_id == 1


# ---------------------------------------------------------------------------
# 7. Calibration
# ---------------------------------------------------------------------------

def test_calibration_default_settings_land_in_expected_range():
    tokens, steps = make_watermarked_corpus(N_STEPS, KEY, ROUNDS, WINDOW, ENTROPY_MIN, MIN_RATIO, seed=1)
    result = wc.compute_detection(wc.sequence_to_pairs(tokens, WINDOW), KEY, ROUNDS, P_THRESHOLD)

    watermarked_fraction = sum(1 for s in steps if s.watermarked) / len(steps)

    # Mean G should be nudged above 0.5, not slammed to 1 -- an ungated
    # scheme would produce something close to 1.0 here.
    assert 0.5 < result.mean_g < 0.7
    assert 0.1 < watermarked_fraction < 0.9
    assert result.p_value < P_THRESHOLD
    assert result.verdict is True


# ---------------------------------------------------------------------------
# 8. Tournament mechanics
# ---------------------------------------------------------------------------

def test_tournament_champion_is_global_bit_sum_max_regardless_of_seeding():
    candidates = [
        wc.Candidate(token_id=1, prob=0.05, bit_sum=3),
        wc.Candidate(token_id=2, prob=0.40, bit_sum=9),   # true max
        wc.Candidate(token_id=3, prob=0.30, bit_sum=7),
        wc.Candidate(token_id=4, prob=0.15, bit_sum=8),
        wc.Candidate(token_id=5, prob=0.10, bit_sum=2),
    ]
    champion, matches = wc.run_tournament(candidates)
    assert champion.token_id == 2
    assert len(matches) >= 1


def test_tournament_tie_breaks_on_probability():
    a = wc.Candidate(token_id=1, prob=0.2, bit_sum=5)
    b = wc.Candidate(token_id=2, prob=0.6, bit_sum=5)
    champion, matches = wc.run_tournament([a, b])
    assert champion.token_id == 2
    assert matches[0].reason == "tie on bit-sum, higher probability"
