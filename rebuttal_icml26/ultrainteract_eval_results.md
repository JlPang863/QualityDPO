# UltraInteract Evaluation Results

## Eurus-7B-SFT + LoRA (UltraInteract Math_CoT 20k)

| Method | GSM8K | MATH | ASDiv | GSM+ | MathQA | Minerva Math | MMLU-Pro Math | BBH | ARC | **Avg** |
|---|---|---|---|---|---|---|---|---|---|---|
| *SFT (ref.)* | *61.3±1.3* | *0.8±0.1* | *13.9±0.7* | *42.1±0.5* | *38.6±0.9* | *17.6±0.5* | *23.5±1.2* | *57.5±0.5* | *57.6±1.4* | *34.77* |
| DPO | 27.4±1.2 | **2.8±0.2** | **25.3±0.9** | 14.0±0.3 | 32.9±0.9 | 9.1±0.4 | 7.3±0.7 | 17.9±0.4 | 55.6±1.5 | 21.36 |
| SimPO | 47.0±1.4 | 0.5±0.1 | 3.0±0.4 | 27.1±0.4 | 34.1±0.9 | 9.4±0.4 | 11.3±0.9 | 12.2±0.4 | 56.1±1.5 | 22.31 |
| SelectiveDPO | **61.8±1.3** | 1.6±0.2 | 19.1±0.8 | **41.2±0.5** | 36.0±0.9 | 13.3±0.5 | 18.7±1.1 | 37.4±0.5 | 57.4±1.4 | 31.84 |
| MixDPO | 61.1±1.3 | 0.5±0.1 | 17.2±0.8 | 40.6±0.5 | **36.4±0.9** | **13.3±0.5** | **18.7±1.1** | **45.4±0.6** | **58.3±1.4** | **32.39** |

### Selected tasks (6 tasks: GSM8K, MathQA, Minerva Math, MMLU-Pro Math, BBH, ARC)

| Method | GSM8K | MathQA | Minerva Math | MMLU-Pro Math | BBH | ARC | **Avg** |
|---|---|---|---|---|---|---|---|
| *SFT (ref.)* | *61.3±1.3* | *38.6±0.9* | *17.6±0.5* | *23.5±1.2* | *57.5±0.5* | *57.6±1.4* | *42.69* |
| DPO | 27.4±1.2 | 32.9±0.9 | 9.1±0.4 | 7.3±0.7 | 17.9±0.4 | 55.6±1.5 | 25.03 |
| SimPO | 47.0±1.4 | 34.1±0.9 | 9.4±0.4 | 11.3±0.9 | 12.2±0.4 | 56.1±1.5 | 28.36 |
| SelectiveDPO | **61.8±1.3** | 36.0±0.9 | 13.3±0.5 | 18.7±1.1 | 37.4±0.5 | 57.4±1.4 | 37.43 |
| MixDPO | 61.1±1.3 | **36.4±0.9** | **13.3±0.5** | **18.7±1.1** | **45.4±0.6** | **58.3±1.4** | **38.87** |

- MathQA: acc_norm reported
- BBH: exact_match (CoT fewshot, get-answer)
- ARC: acc_norm on arc_challenge
- Minerva Math: math_verify metric reported
- MMLU-Pro Math: exact_match (custom-extract) reported

<!-- ## LLaMA-3-8B-SFT Full FT (UltraInteract Math_CoT 55k)

| Method | GSM8K | MATH | ASDiv | GSM+ | **Avg** |
|---|---|---|---|---|---|
| SFT (baseline) | **50.87** | **0.92** | 4.47 | **32.22** | **22.12** |
| DPO | 40.56 | 0.26 | **11.06** | 25.75 | 19.41 |
| SelectiveDPO | 33.59 | 0.10 | 3.34 | 20.35 | 14.35 |
| MixDPO | 20.17 | 0.02 | 4.56 | 12.48 | 9.31 | -->

## Training Configurations

### Eurus-7B LoRA (shared settings)
- Base model: openbmb/Eurus-7b-SFT
- LoRA: r=64, alpha=128, dropout=0.05
- lr=5e-6, beta=0.01, 1 epoch, batch=4*4GPUs*8=128 effective
- max_length=2048, max_prompt_length=1536
- precompute_ref_log_probs=true

### Method-specific differences (Eurus-7B LoRA)

| Config | DPO | SelectiveDPO | MixDPO |
|---|---|---|---|
| Dataset | unsorted_20k | sorted_20k | sorted_20k |
| Dataset fraction | 1.0 (20k) | 0.5 (10k, top margin) | 1.0 (20k) |
| loss_type | sigmoid | sigmoid | noisy-tolerant-4-6-flag |
| label_smoothing | 0.0 | 0.0 | 0.1 |
| Training steps | 156 | 78 | 156 |

### LLaMA-3-8B Full FT (shared settings)
- Base model: princeton-nlp/Llama-3-Base-8B-SFT
- lr=5e-7, beta=0.01, 1 epoch, batch=5*4GPUs*8=160 effective
- max_length=1024, max_prompt_length=512
- precompute_ref_log_probs=false

## Training Dynamics (Eurus-7B LoRA, from wandb)

| Metric | DPO | MixDPO | SelectiveDPO |
|---|---|---|---|
| Final loss | 1.621 | 1.862 | 1.240 |
| Reward margin (end) | 2.22 | 0.52 | 4.10 |
| Reward margin (peak) | 6.03 | 1.99 | 4.41 |
| Reward accuracy (end) | 0.75 | 0.73 | 0.99 |
| Chosen logps (end) | -389 | -279 | -254 |
| Rejected logps (end) | -634 | -304 | -683 |
| Grad norm (end) | 143.9 | 32.6 | 12.0 |
| Grad norm (peak) | 257.9 | 32.6 | 19.9 |

## MixDPO Bug Analysis

**Root cause**: `is_difficult` flag is silently dropped before reaching the loss function.

1. `_signature_columns` (dpo_trainer.py:858) does not include `is_difficult`
   - `remove_unused_columns=True` strips it from the dataset
2. `DataCollatorForPreference.torch_call()` (dpo_trainer.py:158-206) does not forward `is_difficult`
3. Loss function fallback (dpo_trainer.py:1602): `sel_labels = torch.zeros_like(logits)` always executes
   - All 20k samples (including 2k noisy ones) get standard DPO loss

**Consequence**: MixDPO = DPO + label_smoothing=0.1 + sorted data order (not the intended design).

### Dataset noise analysis (20k sorted dataset)
- is_difficult=0: 18,000 samples (90%), mean margin=14.98, 0 negative margins
- is_difficult=1: 2,000 samples (10%), mean margin=-2.63, **68.4% negative margins**
- Negative margin = reward model rates "rejected" higher than "chosen"
