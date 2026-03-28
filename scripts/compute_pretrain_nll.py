"""
Compute per-sample NLL of the SFT model on easy/middle/difficult subsets.
Usage: python scripts/compute_pretrain_nll.py

Outputs: rebuttal_icml26/figures/nll_distribution.png
"""

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
from tqdm import tqdm


def compute_nll_for_response(model, tokenizer, prompt, response, max_length=1024, device="cuda"):
    """Compute NLL of a response given a prompt."""
    # Tokenize prompt + response
    prompt_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length // 2)["input_ids"]
    full_text = prompt + response
    full_ids = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].to(device)

    prompt_len = prompt_ids.shape[1]

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    # Only compute NLL on the response tokens (not the prompt)
    shift_logits = logits[:, prompt_len - 1:-1, :].contiguous()
    shift_labels = full_ids[:, prompt_len:].contiguous()

    if shift_labels.shape[1] == 0:
        return None

    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), reduction="mean")
    return loss.item()


def get_text_from_messages(messages):
    """Extract text from chat messages format."""
    if isinstance(messages, list):
        return " ".join([m.get("content", "") for m in messages if isinstance(m, dict)])
    return str(messages)


def main():
    # Load model
    model_name = "princeton-nlp/Llama-3-Base-8B-SFT"
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    # Load dataset
    print("Loading dataset...")
    ds = load_dataset("jlpang888/ultrafeedback_sorted_score_diff")["train"]

    # Split into easy/middle/difficult by score_diff
    score_diffs = [ds[i]["score_chosen"] - ds[i]["score_rejected"] for i in range(len(ds))]
    total = len(ds)

    # Use same splits as paper: easy (top 1/3), middle (mid 1/3), difficult (bottom 1/3)
    third = total // 3
    easy_indices = list(range(0, third))
    middle_indices = list(range(third, 2 * third))
    difficult_indices = list(range(2 * third, total))

    subsets = {
        "Easy": easy_indices,
        "Middle": middle_indices,
        "Difficult": difficult_indices,
    }

    # Sample to keep it manageable (500 per subset)
    sample_size = 500
    results = {}

    for subset_name, indices in subsets.items():
        np.random.seed(42)
        sampled = np.random.choice(indices, min(sample_size, len(indices)), replace=False)

        chosen_nlls = []
        rejected_nlls = []

        print(f"\nComputing NLL for {subset_name} ({len(sampled)} samples)...")
        for idx in tqdm(sampled):
            row = ds[int(idx)]
            prompt = row["prompt"]

            chosen_text = get_text_from_messages(row["chosen"])
            rejected_text = get_text_from_messages(row["rejected"])

            c_nll = compute_nll_for_response(model, tokenizer, prompt, chosen_text)
            r_nll = compute_nll_for_response(model, tokenizer, prompt, rejected_text)

            if c_nll is not None and r_nll is not None:
                chosen_nlls.append(c_nll)
                rejected_nlls.append(r_nll)

        results[subset_name] = {
            "chosen": chosen_nlls,
            "rejected": rejected_nlls,
        }

        print(f"  Chosen NLL:   mean={np.mean(chosen_nlls):.3f}, std={np.std(chosen_nlls):.3f}")
        print(f"  Rejected NLL: mean={np.mean(rejected_nlls):.3f}, std={np.std(rejected_nlls):.3f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors_chosen = "#2166ac"
    colors_rejected = "#d6604d"

    for ax, (subset_name, data) in zip(axes, results.items()):
        ax.hist(data["chosen"], bins=30, alpha=0.6, color=colors_chosen, label="Chosen", density=True)
        ax.hist(data["rejected"], bins=30, alpha=0.6, color=colors_rejected, label="Rejected", density=True)
        ax.set_title(f"{subset_name} Pairs", fontsize=13, fontweight="bold")
        ax.set_xlabel("NLL (per token)", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    plt.savefig("rebuttal_icml26/figures/nll_distribution.png", dpi=200, bbox_inches="tight")
    plt.savefig("rebuttal_icml26/figures/nll_distribution.pdf", bbox_inches="tight")
    print("\nSaved to rebuttal_icml26/figures/nll_distribution.png")


if __name__ == "__main__":
    main()
