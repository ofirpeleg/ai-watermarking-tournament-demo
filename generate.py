#!/usr/bin/env python
"""
generate.py -- given a --prompt, generates ONE text: the plain greedy
control text by default, or the watermarked text if run with --watermarked.
Run it twice (same prompt) to get both halves of a comparison. Writes the
text plus a report file to runs/.

This is the only program in this project that loads the model. detect.py
never does.
"""

from __future__ import annotations

import argparse
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from watermark_config import MODEL_NAME, load_settings
import watermark_core as wc

RUNS_DIR = "runs"
MAX_PROMPT_WORDS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a normal (control) or watermarked text from a prompt."
    )
    parser.add_argument(
        "--watermarked", "--watermark", action="store_true", dest="watermarked",
        help="generate the watermarked text (tournament sampling) instead of the plain greedy control text",
    )
    parser.add_argument(
        "--model", default=None,
        help=f"override the model to load (default: {MODEL_NAME})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=None, dest="max_tokens",
        help="cap on generated tokens (default: WATERMARK_MAX_TOKENS). The model may stop earlier on its own.",
    )
    parser.add_argument(
        "--prompt", required=True,
        help="the prompt to generate from",
    )
    parser.add_argument(
        "--outputfile", default=None,
        help="base filename to write into runs/ instead of the auto timestamp name "
             "(e.g. --outputfile withwatermark writes runs/withwatermark.txt etc.)",
    )
    return parser.parse_args()


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Prompt validation
# ---------------------------------------------------------------------------

def validate_prompt(prompt: str) -> str:
    prompt = prompt.strip()
    if not prompt:
        raise SystemExit("--prompt cannot be empty.")
    word_count = len(prompt.split())
    if word_count > MAX_PROMPT_WORDS:
        raise SystemExit(f"--prompt is {word_count} words; the limit is {MAX_PROMPT_WORDS}.")
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def build_prompt_ids(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    try:
        # Qwen3's chat template supports enable_thinking; disabling it keeps
        # the output a plain answer instead of a <think>...</think> block,
        # which is what we actually want to watermark. Harmless to pass on
        # tokenizers whose template ignores it.
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt",
            return_dict=True, enable_thinking=False,
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        )
    return encoded["input_ids"]


def generate_normal(model, tokenizer, prompt_ids: torch.Tensor, max_new_tokens: int) -> list[int]:
    with torch.no_grad():
        out = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, prompt_ids.shape[1]:].tolist()


def entropy_bits(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(probs * p.log2()).sum().item())


def generate_watermarked(
    model, tokenizer, prompt_ids: torch.Tensor, max_new_tokens: int, settings,
    device: str = "cpu",
) -> tuple[list[int], list[wc.StepResult]]:
    generated: list[int] = []
    steps: list[wc.StepResult] = []

    cur_input = prompt_ids
    past = None
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=cur_input, past_key_values=past, use_cache=True)
        logits = out.logits[0, -1, :]
        past = out.past_key_values

        probs = torch.softmax(logits, dim=-1)

        h = entropy_bits(probs)
        topk_probs, topk_ids = torch.topk(probs, settings.top_k)
        topk = [(int(tid), float(p)) for tid, p in zip(topk_ids, topk_probs)]

        step = wc.decide_step(
            topk=topk,
            entropy=h,
            generated_so_far=generated,
            key=settings.key,
            rounds=settings.rounds,
            window=settings.context_window,
            entropy_min=settings.entropy_min,
            min_ratio=settings.min_ratio,
        )
        steps.append(step)
        generated.append(step.winner_id)

        if step.winner_id == eos_id:
            break
        cur_input = torch.tensor([[step.winner_id]], device=device)

    return generated, steps


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def tok_repr(tokenizer, token_id: int) -> str:
    return repr(tokenizer.decode([token_id]))


def build_normal_report(model_name: str, settings, prompt: str, normal_text: str, max_new_tokens: int) -> str:
    lines = []
    lines.append("=" * 78)
    lines.append("TOURNAMENT WATERMARKING -- RUN REPORT (normal / control text)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Model:            {model_name}")
    lines.append(f"Max tokens:       {max_new_tokens}  (a cap, not a target -- the model stops on its own)")
    lines.append("")
    lines.append(
        "This is the control text: plain greedy decoding. None of the watermarking\n"
        "settings (key, rounds, gates, context window) do anything here -- they only\n"
        "apply to `python generate.py --watermarked`. Generate that counterpart with\n"
        "the same prompt to get something worth comparing this against."
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("PROMPT")
    lines.append("-" * 78)
    lines.append(prompt)
    lines.append("")
    lines.append("-" * 78)
    lines.append("NORMAL TEXT (greedy, no watermark)")
    lines.append("-" * 78)
    lines.append(normal_text)
    lines.append("")
    return "\n".join(lines) + "\n"


def build_watermarked_report(
    model_name: str, tokenizer, settings, prompt: str, watermarked_text: str,
    steps: list[wc.StepResult], max_new_tokens: int,
) -> tuple[str, dict]:
    n_total = len(steps)
    n_watermarked = sum(1 for s in steps if s.watermarked)
    n_gate_entropy = sum(1 for s in steps if s.gate == "entropy")
    n_gate_viability = sum(1 for s in steps if s.gate == "viability")
    avg_entropy = sum(s.entropy for s in steps) / n_total if n_total else 0.0
    avg_viable = (
        sum(len(s.viable) for s in steps if s.watermarked) / n_watermarked
        if n_watermarked else 0.0
    )

    pairs = wc.sequence_to_pairs([s.winner_id for s in steps], settings.context_window)
    all_g = [wc.g_score(settings.key, settings.rounds, ctx, tid) for ctx, tid in pairs]
    mean_g_all = sum(all_g) / len(all_g) if all_g else 0.0
    wm_g = [s.winner_g for s in steps if s.watermarked]
    mean_g_wm = sum(wm_g) / len(wm_g) if wm_g else 0.0

    lines = []
    lines.append("=" * 78)
    lines.append("TOURNAMENT WATERMARKING -- RUN REPORT (watermarked text)")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"Model:            {model_name}")
    lines.append(f"Key:              {settings.key}")
    lines.append(f"Max tokens:       {max_new_tokens}  (a cap, not a target -- the model stops on its own)")
    lines.append(f"Top-K:            {settings.top_k}")
    lines.append(f"Entropy min:      {settings.entropy_min} bits")
    lines.append(f"Min ratio:        {settings.min_ratio}")
    lines.append(f"Rounds (m):       {settings.rounds}")
    lines.append(f"Context window:   {settings.context_window} generated tokens (never the prompt)")
    lines.append(f"Significance threshold (p must be below): {settings.p_threshold}")
    lines.append("")
    lines.append("-" * 78)
    lines.append("PROMPT")
    lines.append("-" * 78)
    lines.append(prompt)
    lines.append("")
    lines.append("-" * 78)
    lines.append("WATERMARKED TEXT")
    lines.append("-" * 78)
    lines.append(watermarked_text)
    lines.append("")
    lines.append("-" * 78)
    lines.append("SUMMARY")
    lines.append("-" * 78)
    lines.append(f"Tokens generated:            {n_total}")
    lines.append(
        f"Steps that ran a tournament:  {n_watermarked} "
        f"({(n_watermarked / n_total * 100 if n_total else 0):.1f}%)"
    )
    lines.append(f"Skipped by entropy gate:     {n_gate_entropy}")
    lines.append(f"Skipped by viability gate:   {n_gate_viability}")
    lines.append(f"Average entropy:             {avg_entropy:.4f} bits")
    lines.append(f"Average viable candidates:   {avg_viable:.2f} (on tournament steps)")
    lines.append(f"Mean G (whole text):         {mean_g_all:.4f}")
    lines.append(f"Mean G (watermarked steps):  {mean_g_wm:.4f}")
    approx_sigma = 0.5 / (n_total * settings.rounds) ** 0.5 if n_total else float("nan")
    lines.append(
        f"Approx. detection sigma:    {approx_sigma:.6f}  "
        f"(= 0.5 / sqrt(tokens * rounds); more tokens -> lower sigma -> sharper z)"
    )
    lines.append("")
    lines.append("-" * 78)
    lines.append("PER-TOKEN BREAKDOWN")
    lines.append("-" * 78)
    lines.append(
        "Skipped steps get one compact line. Tournament steps get the full "
        "candidate table + match log."
    )
    lines.append("")

    for i, s in enumerate(steps):
        tok = tok_repr(tokenizer, s.winner_id)
        if not s.watermarked:
            reason = "entropy gate" if s.gate == "entropy" else "viability gate"
            lines.append(f"[{i:4d}] SKIPPED ({reason}, H={s.entropy:.3f} bits) -> {tok}")
            continue

        lines.append(f"[{i:4d}] TOURNAMENT  H={s.entropy:.3f} bits")
        lines.append(f"       candidates (top-{settings.top_k}, m={settings.rounds} bits each):")
        for c in s.all_topk:
            if c in s.viable:
                lines.append(
                    f"         {tok_repr(tokenizer, c.token_id):>16}  p={c.prob:.4f}  "
                    f"bit_sum={c.bit_sum}/{settings.rounds}  g={c.bit_sum / settings.rounds:.3f}  [entered]"
                )
            else:
                lines.append(
                    f"         {tok_repr(tokenizer, c.token_id):>16}  p={c.prob:.4f}  [excluded: below min_ratio]"
                )
        lines.append("       matches:")
        for m in s.matches:
            lines.append(
                f"         {tok_repr(tokenizer, m.a_token_id)}(bs={m.a_bit_sum}) vs "
                f"{tok_repr(tokenizer, m.b_token_id)}(bs={m.b_bit_sum}) "
                f"-> winner {tok_repr(tokenizer, m.winner_token_id)} ({m.reason})"
            )
        lines.append(f"       WINNER: {tok}  (g={s.winner_g:.3f})")
        lines.append("")

    report_text = "\n".join(lines) + "\n"

    summary = {
        "tokens_generated": n_total,
        "steps_watermarked": n_watermarked,
        "steps_watermarked_pct": (n_watermarked / n_total * 100 if n_total else 0.0),
        "steps_skipped_entropy_gate": n_gate_entropy,
        "steps_skipped_viability_gate": n_gate_viability,
        "avg_entropy": avg_entropy,
        "avg_viable_candidates": avg_viable,
        "mean_g_all": mean_g_all,
        "mean_g_watermarked_steps": mean_g_wm,
    }
    return report_text, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    settings = load_settings()
    model_name = args.model or MODEL_NAME
    max_new_tokens = args.max_tokens if args.max_tokens is not None else settings.max_tokens

    prompt = validate_prompt(args.prompt)

    device = pick_device()
    print(f"\nLoading {model_name} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32).to(device)
    model.eval()

    prompt_ids = build_prompt_ids(tokenizer, prompt).to(device)

    os.makedirs(RUNS_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = args.outputfile if args.outputfile else f"run_{ts}"

    if args.watermarked:
        print("Generating watermarked text (tournament sampling) ...")
        wm_ids, steps = generate_watermarked(model, tokenizer, prompt_ids, max_new_tokens, settings, device=device)
        watermarked_text = tokenizer.decode(wm_ids, skip_special_tokens=True)

        report_text, summary = build_watermarked_report(
            model_name, tokenizer, settings, prompt, watermarked_text, steps, max_new_tokens
        )

        wm_path = os.path.join(RUNS_DIR, f"{stem}.txt" if args.outputfile else f"{stem}_watermarked.txt")
        report_path = os.path.join(RUNS_DIR, f"{stem}_report.txt")

        with open(wm_path, "w") as f:
            f.write(watermarked_text)
        with open(report_path, "w") as f:
            f.write(report_text)

        if os.environ.get("WATERMARK_PRINT_REPORT"):
            print("\n" + report_text)

        print("\n" + "-" * 60)
        print("Done. (watermarked)")
        print(
            f"Watermarking Summary: {summary['tokens_generated']} steps used tournament sampling; "
            f"{summary['steps_watermarked']} had multiple candidates. "
            f"Average token entropy {summary['avg_entropy']}"
        )
        print(f"Tokens generated:            {summary['tokens_generated']}")
        print(
            f"Steps that ran a tournament:  {summary['steps_watermarked']} "
            f"({summary['steps_watermarked_pct']:.1f}%)"
        )
        print(f"Mean G (whole text):         {summary['mean_g_all']:.4f}")
        print("\nFiles written:")
        for p in (wm_path, report_path):
            print(f"  {p}")
        print(f"\nNext: python detect.py {wm_path}")
        print(
            "(Run generate.py again with the same --prompt, without --watermarked, "
            "to get a control text to compare it against.)"
        )

    else:
        print("Generating normal text (greedy, no watermark) ...")
        normal_ids = generate_normal(model, tokenizer, prompt_ids, max_new_tokens)
        normal_text = tokenizer.decode(normal_ids, skip_special_tokens=True)

        report_text = build_normal_report(model_name, settings, prompt, normal_text, max_new_tokens)

        normal_path = os.path.join(RUNS_DIR, f"{stem}.txt" if args.outputfile else f"{stem}_normal.txt")
        report_path = os.path.join(RUNS_DIR, f"{stem}_report.txt")

        with open(normal_path, "w") as f:
            f.write(normal_text)
        with open(report_path, "w") as f:
            f.write(report_text)

        if os.environ.get("WATERMARK_PRINT_REPORT"):
            print("\n" + report_text)

        print("\n" + "-" * 60)
        print("Done. (normal / control)")
        print(f"Tokens generated:            {len(normal_ids)}")
        print("\nFiles written:")
        for p in (normal_path, report_path):
            print(f"  {p}")
        print("\nNext: run generate.py again with the same --prompt plus --watermarked, to get something to compare this against")
        print(f"Or just check this file's own stats: python detect.py {normal_path}")


if __name__ == "__main__":
    main()
