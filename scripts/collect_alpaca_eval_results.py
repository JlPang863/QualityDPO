"""
Collect AlpacaEval results for all models, including std/CI.
Usage: python scripts/collect_alpaca_eval_results.py

Reads from /home/jlpang/alpaca_eval/model_outputs_cl/
"""

import pandas as pd
import os
from glob import glob

root_path = "/home/jlpang/alpaca_eval/model_outputs_cl"
judge_model = "alpaca_eval_gpt4.1"

# Collect all results
results = []
for model_dir in sorted(os.listdir(root_path)):
    leaderboard_path = os.path.join(root_path, model_dir, judge_model, "leaderboard.csv")
    if os.path.exists(leaderboard_path):
        try:
            df = pd.read_csv(leaderboard_path)
            df["model"] = model_dir
            results.append(df)
        except Exception as e:
            print(f"Error reading {leaderboard_path}: {e}")

if not results:
    print("No results found!")
    exit()

merged = pd.concat(results, ignore_index=True)

# Select key columns
cols = ["model", "length_controlled_winrate", "lc_standard_error", "discrete_win_rate", "standard_error", "avg_length", "n_total"]
available_cols = [c for c in cols if c in merged.columns]
merged = merged[available_cols]

# Round
for c in ["length_controlled_winrate", "discrete_win_rate", "standard_error", "lc_standard_error"]:
    if c in merged.columns:
        merged[c] = merged[c].round(2)
if "avg_length" in merged.columns:
    merged["avg_length"] = merged["avg_length"].round(0).astype(int)

# Print all results
print("=" * 120)
print("ALL MODELS")
print("=" * 120)
print(merged.to_string(index=False, max_colwidth=60))

# Print Qwen results separately
print("\n" + "=" * 120)
print("QWEN 2.5-7B MODELS")
print("=" * 120)
qwen = merged[merged["model"].str.contains("qwen", case=False)]
print(qwen.to_string(index=False, max_colwidth=60))

# Print LLaMA results separately
print("\n" + "=" * 120)
print("LLAMA-3-8B MODELS")
print("=" * 120)
llama = merged[merged["model"].str.contains("llama-3-8b", case=False)]
print(llama.to_string(index=False, max_colwidth=60))

# Save to CSV
output_path = "rebuttal_icml26/alpaca_eval_all_results.csv"
merged.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")
