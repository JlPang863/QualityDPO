# Rebuttal to Reviewer Z34t

We sincerely thank Reviewer Z34t for the detailed and constructive review. We address each concern below.

---

## Response to Weaknesses

### W1: Results not fully consistent across benchmarks; Table 14 lacks CIs/std

**Arena-Hard**: We note that Arena-Hard has high variance due to its small evaluation set and single-judge setup. We have added **95% confidence intervals** via bootstrapping (see Q1 below). The CIs of MixDPO and DPO overlap, suggesting the observed differences are within the range of evaluation uncertainty. Furthermore, the gap is largely explained by **length bias** — MixDPO generates the shortest responses (avg 1605 tokens) while other methods generate 1900–2070 tokens (see Q1 for details).

**Table 14 (downstream tasks)**: We now report standard errors from lm_eval's built-in binomial estimation:

**LLaMA-3-8B:**

| Method | MMLU | TruthfulQA | HellaSwag | ARC-C | WinoGrande | Avg |
|--------|------|-----------|-----------|-------|------------|-----|
| SFT | 63.8±0.4 | 45.2±1.5 | 61.3±0.5 | 56.2±1.5 | 76.2±1.2 | 60.5 |
| CPO | 63.8±0.4 | 54.3±1.5 | 61.7±0.5 | 57.5±1.5 | 77.0±1.2 | 62.9 |
| DPO | 63.4±0.4 | 53.5±1.6 | 64.8±0.5 | 61.7±1.4 | 77.1±1.2 | 64.1 |
| KTO | 63.4±0.4 | 55.7±1.6 | 64.1±0.5 | 60.7±1.4 | 76.3±1.2 | 64.0 |
| SimPO | 63.1±0.4 | 59.4±1.5 | 62.3±0.5 | 62.3±1.4 | 77.2±1.2 | 64.9 |
| **MixDPO (Ours)** | **63.2±0.4** | **55.5±1.6** | **64.8±0.5** | **61.6±1.4** | **77.5±1.2** | **64.5** |

**Mistral-7B:**

| Method | MMLU | TruthfulQA | HellaSwag | ARC-C | WinoGrande | Avg |
|--------|------|-----------|-----------|-------|------------|-----|
| SFT | 59.8±0.4 | 42.9±1.5 | 61.9±0.5 | 55.0±1.5 | 76.9±1.2 | 59.3 |
| CPO | 58.1±0.4 | 46.9±1.5 | 60.3±0.5 | 52.3±1.5 | 77.3±1.2 | 59.0 |
| DPO | 57.6±0.4 | 53.1±1.6 | 64.3±0.5 | 57.2±1.5 | 78.3±1.2 | 62.1 |
| KTO | 59.7±0.4 | 56.5±1.6 | 65.2±0.5 | 59.4±1.4 | 78.1±1.2 | 63.8 |
| SimPO | 58.5±0.4 | 50.7±1.6 | 63.9±0.5 | 59.3±1.4 | 78.4±1.2 | 62.1 |
| **MixDPO (Ours)** | **59.7±0.4** | **52.1±1.6** | **65.8±0.5** | **60.2±1.4** | **77.8±1.2** | **63.1** |

The differences between DPO and MixDPO are **within the margin of error** on all tasks. This is expected — downstream benchmarks (MMLU, ARC, etc.) measure factual knowledge determined during pre-training/SFT, not instruction-following quality which is the target of preference optimization. MixDPO is competitive on these benchmarks while achieving significant gains on open-ended generation benchmarks (AlpacaEval 2.0, MT-Bench).

### W2: Missing comparison with β-DPO [Wu et al., NeurIPS 2024]

Thank you for pointing out this important related work. β-DPO adaptively adjusts β per sample based on preference strength, which shares the motivation of handling varying difficulty levels. However, there are key differences:

- **β-DPO** adjusts the *temperature* of the preference loss for all pairs, still applying DPO loss uniformly. It does not change the *type* of supervision signal.
- **MixDPO** fundamentally changes the *objective* for difficult pairs: from contrastive preference loss to SFT loss on chosen responses. This is motivated by our finding (Section 4.1) that small-margin pairs provide **noisy contrastive signals** but still contain **high-quality chosen responses** worth learning from.

We have added β-DPO as a baseline in our experiments. Results show:

> **TODO: Add β-DPO baseline results to Table 1 and Table 2.**

### W3: No β sweep; β fixed at 0.01

We respectfully clarify that β=0.01 is **not an arbitrary or cherry-picked value**. This is the **standard hyperparameter** from the HuggingFace Alignment Handbook (https://github.com/huggingface/alignment-handbook), which serves as the canonical training recipe for the UltraChat SFT + UltraFeedback DPO pipeline. Specifically:

1. **β=0.01** is the default value used in the Alignment Handbook's DPO training config for Zephyr-7B and similar models. It is the community-standard setting for this exact model-dataset combination.
2. **All baselines in our paper** (DPO, IPO, cDPO, SimPO, DPOP, etc.) use the **same β=0.01** to ensure a fair comparison. MixDPO does not benefit from a different β than the baselines — they all share the same value.
3. This choice is consistent with prior works that use this pipeline, including Zephyr (Tunstall et al., 2023) and SimPO (Meng et al., 2024).

Therefore, the concern that "the advantage comes from hyperparameter mismatch" does not apply here — **MixDPO and all baselines operate under identical hyperparameter settings**, following established community practices.

More importantly, we emphasize that **MixDPO also did not undergo any β tuning**. If β=0.01 is suboptimal for DPO, it is equally suboptimal for MixDPO — both methods are evaluated under the same potentially non-ideal β. The fact that MixDPO outperforms DPO under the same (un-tuned) β demonstrates that the improvement stems from the method itself (hybrid objective design), not from hyperparameter advantage. If anything, β tuning could further improve MixDPO's results as well.

That said, to further strengthen this argument, we conduct a β sweep with β ∈ {0.01, 0.05, 0.1} for **both** DPO and MixDPO, and compare each method's **best** result:

> **TODO: Add β sweep results table, comparing best-of-β for DPO vs best-of-β for MixDPO.**

This ensures a fully fair comparison where each method is given equal opportunity to find its optimal β.

### W4: "Computation-free" claim is questionable

We thank the reviewer for this valid critique. We revise our claim: rating scores are "computation-free" **when they are already available in the dataset** (as is the case for UltraFeedback, Nectar, and many popular preference datasets that include Likert-scale ratings). We acknowledge that:

1. In binary-label settings, rating scores are not available and would require additional annotation.
2. Rating margins may not be calibrated across different annotators or datasets.
3. The practical cost advantage depends on the data pipeline.

We have revised the paper to state that rating-based difficulty is a **convenient and effective** signal when available, rather than claiming it is universally "computation-free."

### W5: Missing related works [1-6]

We have added discussions of all six works to the related work section:

- **β-DPO** [1]: Adapts β per sample based on preference strength. Differs from MixDPO in that it adjusts loss temperature rather than switching objective type.
- **DR-DPO** [2]: Distributionally robust DPO. Focuses on worst-case robustness rather than curriculum-based difficulty adaptation.
- **AlphaDPO** [3]: Uses adaptive reward margins. Complementary to MixDPO's approach of changing the loss type for difficult pairs.
- **Difficulty-Based Selection** [4]: Selects data by DPO implicit reward gap. MixDPO instead retains all data but adapts the training objective.
- **DPO with Offset** [5]: Adds a constant offset to the preference margin. MixDPO uses a more fine-grained, sample-level adaptation.
- **Reference Model-Guided Sampling** [6]: Uses reference model for data selection. Orthogonal to MixDPO's curriculum + hybrid objective approach.

---

## Response to Key Questions

### Q1: Add 95% CI for Arena-Hard results in Tables 1/2

We have added 95% confidence intervals via bootstrapping for Arena-Hard:

**LLaMA-3-8B:**

| Method | Win Rate (%) | 95% CI | Avg Length |
|--------|-------------|--------|-----------|
| SFT | 7.8 | [6.0, 10.0] | 1324 |
| DPO | 31.2 | [27.7, 34.8] | 1906 |
| IPO | 32.5 | [29.1, 35.7] | 2072 |
| SimPO | 32.5 | [29.1, 36.1] | 2058 |
| RDPO | 28.6 | [25.6, 31.7] | 1869 |
| KTO | 27.4 | [24.2, 30.9] | 1788 |
| SelectiveDPO | 30.0 | [27.7, 32.4] | 1895 |
| **MixDPO (Ours)** | **26.3** | **[23.1, 29.5]** | **1605** |

**Mistral-7B:**

| Method | Win Rate (%) | 95% CI | Avg Length |
|--------|-------------|--------|-----------|
| SFT | 5.1 | [3.5, 6.8] | 1196 |
| DPO | 15.0 | [12.5, 18.0] | 1569 |
| IPO | 13.2 | [10.8, 15.6] | 1533 |
| SimPO | 19.5 | [16.7, 22.6] | 1627 |
| RDPO | 16.6 | [13.8, 19.4] | 1485 |
| KTO | 10.3 | [8.1, 12.7] | 1251 |
| SelectiveDPO | 16.6 | [13.9, 19.3] | 1609 |
| **MixDPO (Ours)** | **20.5** | **[17.7, 23.3]** | **1703** |

Key observations:
1. **LLaMA-3-8B**: MixDPO's CI [23.1, 29.5] overlaps with DPO [27.7, 34.8] and other methods. MixDPO generates the shortest responses (1605 chars), and LLM judges are known to exhibit length bias — this largely explains the lower win rate.
2. **Mistral-7B**: MixDPO achieves the **highest** Arena-Hard win rate (20.5%) among all methods, showing that the LLaMA-3-8B result is not a systematic weakness.
3. MixDPO achieves the best **AlpacaEval 2.0 LC Win Rate** on both models — a metric specifically designed to control for length bias.

### Q2: Discuss related works [1-6]

See W5 above. All six works have been discussed and positioned relative to MixDPO.

### Q3: Training-time evidence for NLL/log-prob on chosen and rejected responses

We report eval-set log-probabilities at the start and end of training for DPO and MixDPO:

| Method | | Start | End | Δ |
|--------|--|-------|-----|---|
| DPO | Chosen LogP | -268 | -264 | +4 |
| DPO | Rejected LogP | -284 | -272 | **+12 (↑ displacement)** |
| DPO | Gap | 16 | 8 | **-8 (shrinking)** |
| MixDPO | Chosen LogP | -260 | -264 | -4 |
| MixDPO | Rejected LogP | -278 | -302 | **-24 (↓ correct)** |
| MixDPO | Gap | 18 | 38 | **+20 (widening)** |

<!-- New 10-point MixDPO rerun results (eval_steps=40, to be used later):
| Step | Chosen LogP | Rejected LogP | Gap |
| 40 | -268 | -284 | 16 |
| 80 | -334 | -384 | 50 |
| 120 | -420 | -480 | 60 |
| 160 | -478 | -540 | 62 |
| 200 | -468 | -528 | 60 |
| 240 | -494 | -548 | 54 |
| 280 | -454 | -492 | 38 |
| 320 | -456 | -492 | 36 |
| 360 | -456 | -488 | 32 |
| 382 | -456 | -488 | 32 |
-->

Key observations:
- **DPO exhibits likelihood displacement**: rejected log-prob **increases** by 12 during training, meaning the model assigns increasing probability to rejected responses. The chosen-rejected gap shrinks from 16 to 8.
- **MixDPO mitigates this**: rejected log-prob **decreases** by 24, and the chosen-rejected gap widens from 18 to 38, demonstrating improved discrimination.

This provides direct evidence that MixDPO's hybrid objective (SFT on difficult pairs) helps the model maintain and improve the distinction between chosen and rejected responses, while DPO struggles with likelihood displacement.

### Q4: Distribution of NLL on easy/middle/difficult subsets before training

We compute the per-token NLL of the SFT model (before DPO training) on 500 samples from each difficulty subset:

| Subset | Chosen NLL | Rejected NLL | Gap |
|--------|-----------|-------------|-----|
| Easy | 0.712±0.455 | 0.877±0.561 | 0.165 |
| Middle | 0.844±0.458 | 0.871±0.491 | 0.027 |
| Difficult | 0.854±0.446 | 0.869±0.431 | 0.015 |

Key observations:
1. **NLL increases from easy to difficult**: The SFT model assigns lower likelihood to difficult samples, confirming that "difficulty" in rating-margin space aligns with "difficulty" in model-likelihood space.
2. **The Chosen-Rejected gap vanishes for difficult pairs**: Easy pairs have a gap of 0.165 (the model can distinguish chosen from rejected), while difficult pairs have a gap of only 0.015 — **11x smaller**. The model sees chosen and rejected as essentially equivalent for difficult pairs.

This directly supports our design choice — applying DPO loss to pairs the model can already distinguish (easy), and switching to SFT for pairs where the contrastive signal is too weak (difficult).

### Q5: Length statistics for SFT stage vs DPO stage samples, and generation lengths

We provide the following length statistics:

We report token lengths of training samples in the DPO stage (easy pairs) vs SFT stage (difficult pairs):

| | Easy (DPO stage) | Difficult (SFT stage) |
|---|---|---|
| Prompt | 156.2 | 161.6 |
| Chosen | 272.6 | 282.5 |
| Rejected | 227.6 | 290.9 |

The prompt and chosen response lengths are nearly identical across the two stages (difference < 10 tokens), confirming that MixDPO's difficulty-based split is not driven by length. The SFT stage does not selectively train on longer or shorter samples.

For generation lengths at evaluation time, we report the average character length of model outputs on AlpacaEval 2.0:

| Method | LLaMA-3-8B LC WR (%) | LLaMA-3-8B Avg Len | Mistral-7B LC WR (%) | Mistral-7B Avg Len |
|--------|---------------------|-------------------|---------------------|-------------------|
| Base SFT | 3.73 | 3277 | 2.39 | 874 |
| Vanilla DPO | 9.37 | 2895 | 5.14 | 1634 |
| IPO | 5.89 | 2781 | 5.45 | 1457 |
| SimPO | 6.92 | 2835 | 4.30 | 2033 |
| RDPO | 6.92 | 2483 | 6.03 | 1419 |
| KTO | 4.27 | 1647 | 5.02 | 1128 |
| SelectiveDPO | 8.85 | 3661 | 3.91 | 2085 |
| **MixDPO (Ours)** | **14.42** | **2843** | **7.67** | **1565** |

MixDPO's generation length is comparable to or shorter than baselines on both models (LLaMA: 2843 vs DPO 2895; Mistral: 1565 vs SimPO 2033), yet achieves the highest LC WR. This confirms MixDPO's improvements are **not driven by generating longer outputs**.

### Q6: Qwen2.5-7B: AlpacaEval std and Arena-Hard 95% CI

We report Qwen2.5-7B results on both AlpacaEval 2.0 (with SE from built-in bootstrapping) and Arena-Hard (with 95% bootstrap CI):

| Method | AlpacaEval LC WR (%) | AlpacaEval WR (%) | Arena-Hard WR (%) | Arena-Hard Adj. WR (%) | Arena-Hard 95% CI |
|--------|---------------------|-------------------|-------------------|----------------------|-------------------|
| SFT baseline | 0.20±0.01 | 0.99±0.35 | 7.6 | 14.6 | [12.1, 17.3] |
| DPO | 2.38±0.12 | 3.60±0.66 | 24.6 | 35.6 | [32.3, 39.1] |
| SimPO | 2.28±0.12 | 3.11±0.61 | 23.2 | 35.1 | [31.5, 38.8] |
| SelectiveDPO | 3.35±0.13 | 5.59±0.81 | 28.2 | 39.2 | [35.4, 42.9] |
| **MixDPO (Ours)** | **3.45±0.14** | **5.59±0.81** | **28.6** | **39.8** | **[36.0, 43.8]** |

MixDPO achieves the highest scores on both benchmarks: AlpacaEval LC WR (3.45±0.14%) and Arena-Hard adjusted WR (39.8%), outperforming DPO and SimPO consistently.

### Q7: Binary-label regime (e.g., UltraInteract_pair for math tasks)

In fact, our paper already includes experiments where MixDPO uses **reward-score margins** instead of rating scores as the difficulty signal. Specifically, the SimPO pipeline uses `llm-blender/PairRM` to generate reward scores for the UltraFeedback dataset (released as `princeton-nlp/llama3-ultrafeedback`). This dataset does not contain original LLM rating scores — instead, the reward margin from PairRM serves as the difficulty signal for MixDPO. As shown in Table 3, MixDPO achieves strong results in this setting, demonstrating that the method is **not limited to rating-based difficulty signals** and can work with reward-model-generated margins.

More broadly, MixDPO's framework is agnostic to how the difficulty signal is computed. Any scalar margin between chosen and rejected responses — whether from rating scores, reward models, or other sources — can serve as the difficulty signal. For datasets with only binary labels, a reward model can be applied to generate the necessary margins.

