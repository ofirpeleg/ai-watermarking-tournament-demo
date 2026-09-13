#!/usr/bin/env python
"""
detect.py -- standalone statistical detector. Given one or more run files
(.txt or .json), decides whether each one carries the tournament watermark.

MUST NEVER load the model or model weights. A .txt file is re-tokenized with
the tokenizer alone (transformers, no torch needed for that); a .json run
file already has token ids, so nothing beyond scipy is needed for those.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import watermark_core as wc
from watermark_config import MODEL_NAME, load_settings

_tokenizers: dict[str, object] = {}


def get_tokenizer(model_name: str):
    if model_name not in _tokenizers:
        from transformers import AutoTokenizer
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
    return _tokenizers[model_name]


def format_p(p: float) -> str:
    """Plain decimal -- never scientific ('e') notation. Uses just enough
    decimal places to actually show the value (e.g. 0.0001244 stays visible
    instead of rounding to 0.0000000), but caps out at 20 places: float64
    itself only carries about 15-17 significant digits, so beyond that a
    p-value this extreme is already indistinguishable from zero -- more
    zeros wouldn't be showing real information, just padding."""
    if p == 0:
        return f"{0.0:.20f}"
    if p >= 1:
        return f"{p:.7f}"
    decimals = min(20, max(7, -math.floor(math.log10(abs(p))) + 3))
    return f"{p:.{decimals}f}"


class Doc:
    def __init__(self, label: str, token_ids: list[int], key: str, rounds: int, window: int, p_threshold: float,
                 tokenizer_name: str, gen_summary: dict | None = None):
        self.label = label
        self.token_ids = token_ids
        self.key = key
        self.rounds = rounds
        self.window = window
        self.p_threshold = p_threshold
        self.tokenizer_name = tokenizer_name
        self.gen_summary = gen_summary  # generation-time tournament stats, only available from a .json input
        self.result: wc.DetectionResult | None = None


def load_docs_from_file(path: str, args, defaults) -> list[Doc]:
    ext = os.path.splitext(path)[1].lower()
    key = args.key if args.key is not None else defaults.key
    window = args.window if args.window is not None else defaults.context_window
    rounds = args.rounds if args.rounds is not None else defaults.rounds
    p_threshold = args.p_threshold if args.p_threshold is not None else defaults.p_threshold

    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        settings = data.get("settings", {})
        # embedded settings win over env-var defaults, but explicit CLI flags win over everything
        key = args.key if args.key is not None else settings.get("key", key)
        window = args.window if args.window is not None else settings.get("context_window", window)
        rounds = args.rounds if args.rounds is not None else settings.get("rounds", rounds)
        p_threshold = args.p_threshold if args.p_threshold is not None else settings.get("p_threshold", p_threshold)
        tokenizer_name = args.model or data.get("model", MODEL_NAME)

        docs = []
        for section in ("normal", "watermarked"):
            if section in data:
                docs.append(Doc(
                    label=f"{os.path.basename(path)}::{section}",
                    token_ids=data[section]["token_ids"],
                    key=key, rounds=rounds, window=window, p_threshold=p_threshold,
                    tokenizer_name=tokenizer_name,
                    gen_summary=data[section].get("summary"),
                ))
        return docs

    elif ext == ".txt":
        with open(path) as f:
            text = f.read()
        tokenizer_name = args.model or MODEL_NAME
        tokenizer = get_tokenizer(tokenizer_name)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return [Doc(
            label=os.path.basename(path),
            token_ids=token_ids,
            key=key, rounds=rounds, window=window, p_threshold=p_threshold,
            tokenizer_name=tokenizer_name,
        )]

    else:
        raise ValueError(f"Unsupported file type: {path} (expected .txt or .json)")


LEGEND = """\
Field reference:
  Tokens analyzed   distinct (context, token) pairs actually scored (after dedup, unless --no-dedup)
  g=1 / g=0 count   keyed bits, summed across every scored token and all `rounds` bits each, that came out 1 / 0
  Mean g-score      g1 / (g1 + g0) -- the core statistic. 0.5 under no watermark; pushed above 0.5 by one
  expected          the null-hypothesis mean: always 0.5, since an unbiased hash has no reason to favor 1 or 0
  z-score           (Mean g-score - expected) / sigma -- how many standard errors above the null this is
  p-value           one-sided probability of a z this high, or higher, if the text were never watermarked
"""


def print_doc_report(doc: Doc, dedup: bool, show_label: bool):
    r = doc.result
    if show_label:
        print("=" * 60)
        print(f"File: {doc.label}")
    print(f"Tokenizer: {doc.tokenizer_name}")
    print(f"Key: {doc.key!r}   Context window: {doc.window}   Rounds: {doc.rounds}   Dedup: {dedup}")
    print(f"Tokens analyzed:  {r.tokens_analyzed}" + (
        f"  (deduped from {r.deduped_from})" if r.deduped_from else ""
    ))
    print(f"g=1 count:        {r.g1_count}")
    print(f"g=0 count:        {r.g0_count}")
    print(f"Mean g-score:     {r.mean_g:.4f}   (null expectation: {r.expected_mean:.4f})")
    print(f"z-score:          {r.z:.3f}")
    print(f"p-value:          {format_p(r.p_value)}")
    if doc.gen_summary:
        s = doc.gen_summary
        print(
            f"Tournament summary (from generation record): "
            f"{s['steps_watermarked']}/{s['tokens_generated']} steps ran a tournament "
            f"({s['steps_watermarked_pct']:.1f}%)"
        )
    print()
    if r.verdict:
        print(f"Verdict: WATERMARK DETECTED (p < {doc.p_threshold})")
    else:
        print(f"Verdict: no watermark detected")
    print()


def print_comparison(docs: list[Doc]):
    print("=" * 60)
    print("COMPARISON")
    print("=" * 60)
    header = f"{'file':<40} {'N':>7} {'mean g':>8} {'z':>8} {'p':>22}  verdict"
    print(header)
    print("-" * len(header))
    for d in docs:
        r = d.result
        v = "DETECTED" if r.verdict else "no watermark"
        print(f"{d.label:<40} {r.tokens_analyzed:>7} {r.mean_g:>8.4f} {r.z:>8.3f} {format_p(r.p_value):>22}  {v}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Detect tournament watermarking in generated text.")
    parser.add_argument("files", nargs="*", default=[], help=".txt or .json run files")
    parser.add_argument(
        "--input", action="append", default=[], dest="input_files",
        help="alias for a positional file argument; can be repeated (--input a.txt --input b.txt)",
    )
    parser.add_argument("--model", default=None, help=f"override the tokenizer to use for .txt files (default: {MODEL_NAME})")
    parser.add_argument("--key", default=None, help="override the watermark key")
    parser.add_argument("--window", type=int, default=None, help="override the context window")
    parser.add_argument("--rounds", type=int, default=None, help="override the keyed hash round count (m)")
    parser.add_argument("--p-threshold", type=float, default=None, dest="p_threshold", help="override the significance threshold")
    parser.add_argument("--no-dedup", action="store_true", help="disable (context, token) deduplication")
    parser.add_argument("--no-legend", action="store_true", help="skip the field-reference legend")
    args = parser.parse_args()

    all_files = args.files + args.input_files
    if not all_files:
        parser.error("no input files given (pass them positionally or with --input)")

    defaults = load_settings()
    dedup = not args.no_dedup

    if not args.no_legend:
        print(LEGEND)

    docs: list[Doc] = []
    for path in all_files:
        docs.extend(load_docs_from_file(path, args, defaults))

    for doc in docs:
        pairs = wc.sequence_to_pairs(doc.token_ids, doc.window)
        doc.result = wc.compute_detection(pairs, doc.key, doc.rounds, doc.p_threshold, dedup=dedup)
        print_doc_report(doc, dedup, show_label=len(docs) > 1)

    if len(docs) > 1:
        print_comparison(docs)


if __name__ == "__main__":
    main()
