"""
Compute eval-set chosen/rejected log-probabilities for existing models.
Usage: python scripts/compute_eval_logps.py

Reports mean sequence-level sum logps on the eval set for each model.
"""

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json


def get_response_text(messages):
    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content", "")
        return messages[-1].get("content", "") if messages else ""
    return str(messages)


def compute_sequence_logp(model, tokenizer, prompt, response, max_length=1024, device="cuda"):
    # Tokenize prompt and full text from the same tokenization to get accurate prompt_len
    prompt_ids = tokenizer(prompt, truncation=True, max_length=max_length // 2)["input_ids"]
    full_text = prompt + response
    full_ids_list = tokenizer(full_text, truncation=True, max_length=max_length)["input_ids"]

    # Find prompt length within the full tokenization
    # Use the prompt token count as an approximation (BPE boundary effect is minimal)
    prompt_len = len(prompt_ids)
    if len(full_ids_list) <= prompt_len:
        return None

    full_ids = torch.tensor([full_ids_list], device=device)

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits

    shift_logits = logits[:, prompt_len - 1:-1, :]
    shift_labels = full_ids[:, prompt_len:]

    if shift_labels.shape[1] == 0:
        return None

    log_probs = torch.gather(
        shift_logits.log_softmax(-1), dim=2, index=shift_labels.unsqueeze(2)
    ).squeeze(2)

    # Sum log probs (sequence-level, same as DPO trainer)
    return log_probs.sum().item()


def eval_model(model_path, dataset, tokenizer, num_samples=500, device="cuda"):
    print(f"\nLoading model: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    np.random.seed(42)
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    chosen_logps = []
    rejected_logps = []

    for idx in tqdm(indices, desc="Computing logps"):
        row = dataset[int(idx)]
        prompt = row["prompt"]
        chosen_text = get_response_text(row["chosen"])
        rejected_text = get_response_text(row["rejected"])

        c_logp = compute_sequence_logp(model, tokenizer, prompt, chosen_text, device=device)
        r_logp = compute_sequence_logp(model, tokenizer, prompt, rejected_text, device=device)

        if c_logp is not None and r_logp is not None:
            chosen_logps.append(c_logp)
            rejected_logps.append(r_logp)

    del model
    torch.cuda.empty_cache()

    mean_chosen = np.mean(chosen_logps)
    mean_rejected = np.mean(rejected_logps)
    return mean_chosen, mean_rejected


def main():
    # Models to evaluate
    models = {
        # LLaMA-3-8B
        "LLaMA SFT": "princeton-nlp/Llama-3-Base-8B-SFT",
        "LLaMA DPO": "princeton-nlp/Llama-3-Base-8B-SFT-DPO",
        "LLaMA MixDPO": "/mnt/data1/jinlong/CL_DPO_outputs/llama-3-8b-ours4-6-sorted-score-diff-full",
        # Mistral-7B
        "Mistral SFT": "HuggingFaceH4/mistral-7b-sft-beta",
        "Mistral DPO": "princeton-nlp/Mistral-7B-Base-SFT-DPO",
        "Mistral MixDPO": "/mnt/data1/jinlong/CL_DPO_outputs/mistral-7b-ours4-6-sorted-score-diff-new-base-full-lr5",
        # Qwen-2.5-7B
        "Qwen SFT": "AmberYifan/Qwen2.5-7B-sft-ultrachat",
        "Qwen DPO": "/mnt/data1/jinlong/CL_DPO_outputs/qwen-2.5-7b-dpo-full",
        "Qwen MixDPO": "/mnt/data1/jinlong/CL_DPO_outputs/qwen-2.5-7b-ours4-6-sorted-score-diff-full",
    }

    # Load eval dataset
    print("Loading eval dataset...")
    eval_ds = load_dataset("jlpang888/ultrafeedback_sorted_score_diff")["test"]
    print(f"Eval set: {len(eval_ds)} samples")

    # Use LLaMA tokenizer for all (logps are relative, not absolute)
    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/Llama-3-Base-8B-SFT")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    for name, path in models.items():
        try:
            # Use model's own tokenizer if different family
            if "mistral" in name.lower():
                tok = AutoTokenizer.from_pretrained("HuggingFaceH4/mistral-7b-sft-beta")
            elif "qwen" in name.lower():
                tok = AutoTokenizer.from_pretrained("AmberYifan/Qwen2.5-7B-sft-ultrachat")
            else:
                tok = tokenizer

            if tok.pad_token is None:
                tok.pad_token = tok.eos_token

            mean_c, mean_r = eval_model(path, eval_ds, tok, num_samples=500)
            results[name] = {"chosen": mean_c, "rejected": mean_r, "gap": mean_c - mean_r}
            print(f"  {name}: chosen={mean_c:.1f}, rejected={mean_r:.1f}, gap={mean_c - mean_r:.1f}")
        except Exception as e:
            print(f"  {name}: FAILED - {e}")

    # Print summary table
    print("\n" + "=" * 80)
    print("| Model | Chosen LogP | Rejected LogP | Gap |")
    print("|-------|------------|--------------|-----|")

    # Group by model family and compute delta vs SFT
    for family in ["LLaMA", "Mistral", "Qwen"]:
        sft_key = f"{family} SFT"
        dpo_key = f"{family} DPO"
        mix_key = f"{family} MixDPO"

        if sft_key in results and dpo_key in results and mix_key in results:
            sft = results[sft_key]
            dpo = results[dpo_key]
            mix = results[mix_key]

            print(f"\n**{family}:**")
            print(f"| SFT (baseline) | {sft['chosen']:.1f} | {sft['rejected']:.1f} | {sft['gap']:.1f} |")
            print(f"| DPO | {dpo['chosen']:.1f} (Δ={dpo['chosen']-sft['chosen']:+.1f}) | {dpo['rejected']:.1f} (Δ={dpo['rejected']-sft['rejected']:+.1f}) | {dpo['gap']:.1f} (Δ={dpo['gap']-sft['gap']:+.1f}) |")
            print(f"| MixDPO | {mix['chosen']:.1f} (Δ={mix['chosen']-sft['chosen']:+.1f}) | {mix['rejected']:.1f} (Δ={mix['rejected']-sft['rejected']:+.1f}) | {mix['gap']:.1f} (Δ={mix['gap']-sft['gap']:+.1f}) |")

    # Save results
    with open("rebuttal_icml26/eval_logps_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to rebuttal_icml26/eval_logps_results.json")


if __name__ == "__main__":
    main()
