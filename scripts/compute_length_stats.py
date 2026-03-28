"""
Compute token length statistics for DPO stage (easy pairs) vs SFT stage (difficult pairs).
Also computes generation length from AlpacaEval outputs if available.

Usage: python scripts/compute_length_stats.py

Outputs: rebuttal_icml26/figures/length_stats.png + printed table
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
import json
import os
from glob import glob


def get_response_text(messages):
    """Extract assistant response text from chat messages."""
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content", "")
        # fallback: last message
        return messages[-1].get("content", "") if messages else ""
    return str(messages)


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-Base-8B-SFT")

    # Load dataset (sorted by score_diff, easy first)
    print("Loading dataset...")
    ds = load_dataset("jlpang888/ultrafeedback_sorted_score_diff")["train"]
    total = len(ds)

    # Split: easy (score_diff >= 0.5) vs difficult (score_diff < 0.5)
    threshold = 0.5
    easy_idx = []
    difficult_idx = []
    for i in range(total):
        diff = ds[i]["score_chosen"] - ds[i]["score_rejected"]
        if diff >= threshold:
            easy_idx.append(i)
        else:
            difficult_idx.append(i)

    print(f"Easy (DPO stage): {len(easy_idx)} pairs")
    print(f"Difficult (SFT stage): {len(difficult_idx)} pairs")

    # Compute token lengths
    subsets = {"Easy (DPO stage)": easy_idx, "Difficult (SFT stage)": difficult_idx}
    results = {}

    for name, indices in subsets.items():
        chosen_lens = []
        rejected_lens = []
        prompt_lens = []

        # Sample for speed (2000 per subset)
        np.random.seed(42)
        sampled = np.random.choice(indices, min(2000, len(indices)), replace=False)

        print(f"\nTokenizing {name} ({len(sampled)} samples)...")
        for idx in sampled:
            row = ds[int(idx)]

            prompt_tokens = tokenizer(row["prompt"], truncation=False)["input_ids"]
            chosen_text = get_response_text(row["chosen"])
            rejected_text = get_response_text(row["rejected"])
            chosen_tokens = tokenizer(chosen_text, truncation=False)["input_ids"]
            rejected_tokens = tokenizer(rejected_text, truncation=False)["input_ids"]

            prompt_lens.append(len(prompt_tokens))
            chosen_lens.append(len(chosen_tokens))
            rejected_lens.append(len(rejected_tokens))

        results[name] = {
            "prompt": prompt_lens,
            "chosen": chosen_lens,
            "rejected": rejected_lens,
        }

        print(f"  Prompt:   mean={np.mean(prompt_lens):.1f}, median={np.median(prompt_lens):.1f}, std={np.std(prompt_lens):.1f}")
        print(f"  Chosen:   mean={np.mean(chosen_lens):.1f}, median={np.median(chosen_lens):.1f}, std={np.std(chosen_lens):.1f}")
        print(f"  Rejected: mean={np.mean(rejected_lens):.1f}, median={np.median(rejected_lens):.1f}, std={np.std(rejected_lens):.1f}")

    # Print summary table
    print("\n" + "=" * 70)
    print("| Subset | | Mean | Median | Std |")
    print("|--------|----------|------|--------|-----|")
    for name, data in results.items():
        for resp_type in ["prompt", "chosen", "rejected"]:
            vals = data[resp_type]
            print(f"| {name} | {resp_type:8s} | {np.mean(vals):5.1f} | {np.median(vals):6.1f} | {np.std(vals):5.1f} |")
    print("=" * 70)

    # Plot histograms
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (name, data) in zip(axes, results.items()):
        ax.hist(data["chosen"], bins=40, alpha=0.6, color="#2166ac", label="Chosen", density=True)
        ax.hist(data["rejected"], bins=40, alpha=0.6, color="#d6604d", label="Rejected", density=True)
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Token Length", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    os.makedirs("rebuttal_icml26/figures", exist_ok=True)
    plt.savefig("rebuttal_icml26/figures/length_stats.png", dpi=200, bbox_inches="tight")
    plt.savefig("rebuttal_icml26/figures/length_stats.pdf", bbox_inches="tight")
    print("\nSaved to rebuttal_icml26/figures/length_stats.png")

    # Check for AlpacaEval generation outputs
    print("\n--- Generation Length (AlpacaEval outputs) ---")
    alpaca_dirs = glob("/mnt/data1/jinlong/*/alpaca_eval*") + glob("/mnt/data1/jinlong/*/*alpaca*")
    if alpaca_dirs:
        for d in alpaca_dirs[:5]:
            print(f"  Found: {d}")
        print("  (Load these to compute generation lengths)")
    else:
        print("  No AlpacaEval outputs found. Run evaluation first, then rerun this script.")


if __name__ == "__main__":
    main()
