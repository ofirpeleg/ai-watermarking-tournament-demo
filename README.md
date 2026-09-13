# Tournament Watermarking

An implementation of tournament-based AI watermarking: the same core idea
behind Google's SynthID-Text, just small enough that you can read
and actually understand it.

The idea: an LLM almost always has more than one reasonable word it could say next. 
Each candidate is hashed with a secret key into a score; 
the candidates then compete in a tournament, and one winner is emitted as the actual next word. 
To a reader, the text looks completely normal. 
But anyone holding the key can go back later and tell, with a p-value (how unlikely this pattern would be by pure chance), whether a given text came from that model.


This is **not** production-grade (see [Limits](#limits)) — it's built to be
learned from.

## Two programs, on purpose

- **`generate.py`** — loads the model, writes text.
- **`detect.py`** — checks text for the watermark. **It never loads the model.** It only needs a tokenizer and
  `scipy` — no `torch` at all, on purpose, to make the point that detection
  doesn't need the AI stack.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Generate a plain text and a watermarked one from the same prompt:

```bash
python generate.py --prompt "Two sentences about lighthouses."
python generate.py --prompt "Two sentences about lighthouses." --watermarked
```

Each writes 2 files into `runs/` (the text, and a full report), named after
the timestamp (e.g. `run_20260913_161041_watermarked.txt`) unless you name
them yourself with `--outputfile`. Then check the text:

```bash
python detect.py runs/run_<timestamp>_watermarked.txt
```

Only want to run `detect.py` somewhere (e.g. a machine that will never touch
the model)? `pip install -r requirements-detect.txt` instead — no `torch`.

Useful flags: `--model NAME` (swap models), `--max-tokens N`,
`--outputfile NAME` (fixed filename instead of an auto timestamp — 
`--outputfile withwatermark` writes `runs/withwatermark.txt`). Detection is
deterministic and works on plain `.txt` files too, no model needed:

```bash
python detect.py runs/run_<ts>_watermarked.txt --key wrong-key   # wrong key -> not detected
python detect.py runs/run_<ts>_watermarked.txt --window 9        # wrong window -> not detected
```

## How it works

Per generated token:

```
logits -> softmax -> entropy of this step
  |
  gate 1: model very confident?  -> keep its own top pick, no watermark
  |
  top-K candidates by probability
  |
  gate 2: drop anything far less likely than the leader
          fewer than 2 left?     -> no real choice, keep the top pick
  |
  score each survivor with m keyed hash bits, sum them
  |
  tournament: highest score wins (ties -> higher probability)
  |
  winner gets emitted, becomes context for the next token
```

**logits -> softmax -> entropy:** the model outputs one raw score (a
"logit") per possible next word; softmax turns those into real probabilities
that sum to 1. Entropy then measures how spread out those probabilities
are — near 0 when one word totally dominates (nothing to nudge), higher when
several words are genuinely competitive (room to watermark).

**Bigger models produce lower entropy, on the same prompt.** A larger model
is simply more confident about which word comes next — for common tasks
especially, it's seen enough during training to be fairly sure of itself.
Measured on the same prompt: `Qwen2.5-0.5B-Instruct` averaged ~2.65 bits of
entropy per token, `Qwen2.5-1.5B-Instruct` (3x the parameters) dropped to
~1.59, and a much larger model (~8B class) dropped further still, to ~0.67.
Less entropy means fewer steps clear the entropy gate, which means a
smaller fraction of the text actually gets watermarked — so a bigger model
needs more tokens to reach the same statistical significance as a smaller
one.

The hash: `g_bit(round, context, token) = low bit of SHA256(key | round | context | token)`
— run `m` times per candidate (`m` = `WATERMARK_ROUNDS`) and summed into
that candidate's score. `context` is the last 4 **generated** tokens only. 
Detection just recomputes those same
bits for the token that actually got emitted and checks if they lean 1 more
than a coin flip would.

Two gates, not one, matter here: a real early-version bug let the tournament
pick between `' Name'` (99.5% likely) and `' Names'` (0.18% likely) just
because both were "top-K" — and a run of those produced *"DNS was developed
by Ray Tomsho in 1891."* The fix is gate 2 (`min_ratio`): a candidate has to
be a real contender, not just present in the top-K list.

## Proof it works

**A random creative-writing prompt, watermarked vs. not**, same prompt, same
model ("Write a short story about a dragon who is afraid of heights and
works as a mail carrier."):

```
                  tokens   mean g-score    z       p-value        verdict
plain             400      0.5046       1.00      0.1576512      no watermark
watermarked       311      0.5143       2.75      0.0029449      DETECTED
```

**What's z-score?** It's how many "standard errors" away from 0.5 the
Mean g-score landed — basically, how surprising the result is if there were
no watermark at all. `z ≈ 0` means "exactly what random chance looks like."
`z = 1.00` (the plain text above) is unremarkable — well within normal
noise. `z = 2.75` (the watermarked one) is unlikely enough to happen by
chance that it clears the significance bar. The p-value is that same idea
converted into a probability: "what are the odds of seeing a z this
extreme if there really were no watermark?" (Gate settings control how
extreme these numbers get — see "A few things worth knowing" below.)

**A task with no real choices — counting 1 to 50 — proves the gates work.**
Asked to type out every number with no shortcuts, the model did:

```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, ..., 48, 49, 50
```

and the run reported: *`190 steps used tournament sampling; 0 had multiple
candidates. Average token entropy 0.004`.* Zero tournaments ran — after
"1, 2, 3, 4," there's only one sane next token, so both gates blocked
watermarking on every single step. Mean G came out at 0.4925 — essentially
0.5, exactly as it should when there was nothing to nudge. That's the
system correctly refusing to fake evidence where none exists.

## Config

Everything's an env var:

| Variable | Default | What it does |
|---|---|---|
| `WATERMARK_KEY` | `my-demo-secret-key` | the secret key |
| `WATERMARK_MAX_TOKENS` | `700` | cap on generated tokens, used if `--max-tokens` isn't given. Just a ceiling — the model stops on its own whenever it's done, it doesn't aim for this number |
| `WATERMARK_TOP_K` | `20` | candidate pool size |
| `WATERMARK_ENTROPY_MIN` | `2.0` | min entropy (bits) to even consider watermarking |
| `WATERMARK_MIN_RATIO` | `0.3` | how close to the leader a candidate must be |
| `WATERMARK_ROUNDS` | `30` | keyed hash rounds per token (`m`) |
| `WATERMARK_CONTEXT` | `4` | generated tokens feeding the hash |
| `WATERMARK_P_THRESHOLD` | `0.01` | how sure detection needs to be before saying "yes, watermarked" (p-value must be below this) |

`generate.py` and `detect.py` read these from the same env vars, so as long
as you don't change them between a generate and a detect run, they always
match automatically. Override on the `detect.py` side with
`--key`/`--window`/`--rounds`/`--p-threshold` to demonstrate a mismatch on purpose.

## A few things worth knowing

- **Deduplication matters.** The detector scores each (context, token) pair
  once, even if it repeats. A model looping on one phrase would otherwise
  count the same "evidence" hundreds of times and produce a fake, wildly
  confident detection. `--no-dedup` disables this on purpose, to show the
  failure.
- **Mean g-score is never close to 1.** A watermark with gates leaves most
  of the text untouched (see the counting example above), so it stays near
  0.5 with just enough lean to be statistically obvious over enough tokens.
  Near 1.0 would mean an ungated, brute-force scheme, not a realistic one.
- **More rounds (`m`) barely changes detection strength** — it mostly makes
  Mean G look more subtle/production-like, at the cost of respecting the
  model's own word choice less often.
- **Stricter gates (`WATERMARK_ENTROPY_MIN`, `WATERMARK_MIN_RATIO`) give
  smaller, more readable z/p-values instead of an extreme wall of digits.**
  With loose gates (e.g. `0.5` / `0.1`), most steps get watermarked and the
  numbers get astronomically extreme (`z` in the double digits). The
  defaults here (`2.0` / `0.3`) watermark a smaller, more selective slice of
  steps, landing on more modest, human-readable numbers — at the cost of
  needing more tokens to stay significant on short text.


## Limits

- Not SynthID-Text or any published scheme. a simplified, from-scratch
  reimplementation of the same idea.
- Not robust to paraphrasing or heavy editing: a rewrite changes the
  token context the hash depends on.
- Detector needs the exact key, round count, context window, and tokenizer
  used at generation — get any wrong and it silently fails to detect, rather
  than erroring.

## Tests

```bash
pytest tests/ -v
```

Runs against synthetic probability distributions, not the real model, so
it's fast and deterministic — covers alignment, key/window sensitivity, the
repetition false-positive, and calibration.

## Credits

Modeled after Google's **SynthID-Text** -
[Dathathri et al., *Nature* 634, 818–823 (2024)](https://www.nature.com/articles/s41586-024-08025-4)

The Best Youtube Channel to ever exists - 
[Computerphile's "How Watermarks Track AI Generated Content"](https://www.youtube.com/watch?v=kVXp6UNVPTo),
which several of this project's defaults (`top_k=20`, `context=4`,
`rounds=30`) are matched to.