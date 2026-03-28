"""
Collect downstream task results (with stderr) from lm_eval outputs.
Usage: python scripts/collect_downstream_results.py

Reads from downstream_task_results/
"""

import json
import os
import pandas as pd
from glob import glob

results_dir = "downstream_task_results"

# Tasks we care about (matching Table 14 in paper)
TASKS = {
    "mmlu": "MMLU",
    "truthfulqa_mc2": "TruthfulQA",
    "hellaswag": "HellaSwag",
    "arc_challenge": "ARC-C",
    "gsm8k": "GSM8K",
    "winogrande": "WinoGrande",
}

# Collect results
all_results = []

for model_dir in sorted(os.listdir(results_dir)):
    model_path = os.path.join(results_dir, model_dir)
    if not os.path.isdir(model_path):
        continue

    # Find all result json files
    json_files = glob(os.path.join(model_path, "**", "results_*.json"), recursive=True)
    if not json_files:
        continue

    # Use the latest result file for each task
    task_results = {}
    for jf in sorted(json_files):
        try:
            with open(jf) as f:
                data = json.load(f)
            results = data.get("results", {})
            for task_key, task_name in TASKS.items():
                if task_key in results:
                    acc = results[task_key].get("acc,none") or results[task_key].get("acc_norm,none")
                    stderr = results[task_key].get("acc_stderr,none") or results[task_key].get("acc_norm_stderr,none")
                    if acc is not None:
                        task_results[task_name] = {"acc": acc, "stderr": stderr}
        except Exception as e:
            continue

    if task_results:
        row = {"Model": model_dir}
        accs = []
        for task_name in TASKS.values():
            if task_name in task_results:
                acc = task_results[task_name]["acc"] * 100
                stderr = task_results[task_name]["stderr"] * 100 if task_results[task_name]["stderr"] else 0
                row[task_name] = f"{acc:.1f}±{stderr:.1f}"
                accs.append(acc)
            else:
                row[task_name] = "-"
        row["Avg"] = f"{sum(accs)/len(accs):.1f}" if accs else "-"
        all_results.append(row)

# Create DataFrame
df = pd.DataFrame(all_results)
cols = ["Model"] + list(TASKS.values()) + ["Avg"]
df = df[[c for c in cols if c in df.columns]]

print("=" * 120)
print("DOWNSTREAM TASK RESULTS (acc% ± stderr%)")
print("=" * 120)
print(df.to_string(index=False, max_colwidth=50))

# Save
output_path = "rebuttal_icml26/downstream_task_results.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

# Print markdown table
print("\n\nMarkdown Table:")
print("| Model | " + " | ".join(TASKS.values()) + " | Avg |")
print("|" + "---|" * (len(TASKS) + 2))
for _, row in df.iterrows():
    vals = [str(row.get(c, "-")) for c in cols]
    print("| " + " | ".join(vals) + " |")
