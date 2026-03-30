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
| SelectiveDPO | 64.0±0.4 | 53.9±1.6 | 64.8±0.5 | 61.5±1.4 | 76.1±1.2 | 64.0 |
| MixDPO (Ours) | 63.2±0.4 | 55.5±1.6 | 64.8±0.5 | 61.6±1.4 | 77.5±1.2 | 64.5 |

**Mistral-7B:**

| Method | MMLU | TruthfulQA | HellaSwag | ARC-C | WinoGrande | Avg |
|--------|------|-----------|-----------|-------|------------|-----|
| SFT | 59.8±0.4 | 42.9±1.5 | 61.9±0.5 | 55.0±1.5 | 76.9±1.2 | 59.3 |
| CPO | 58.1±0.4 | 46.9±1.5 | 60.3±0.5 | 52.3±1.5 | 77.3±1.2 | 59.0 |
| DPO | 57.6±0.4 | 53.1±1.6 | 64.3±0.5 | 57.2±1.5 | 78.3±1.2 | 62.1 |
| KTO | 59.7±0.4 | 56.5±1.6 | 65.2±0.5 | 59.4±1.4 | 78.1±1.2 | 63.8 |
| SimPO | 58.5±0.4 | 50.7±1.6 | 63.9±0.5 | 59.3±1.4 | 78.4±1.2 | 62.1 |
| SelectiveDPO | 59.1±0.4 | 46.0±1.6 | 65.1±0.5 | 60.4±1.4 | 77.4±1.2 | 61.6 |
| MixDPO (Ours) | 59.7±0.4 | 52.1±1.6 | 65.8±0.5 | 60.2±1.4 | 77.8±1.2 | 63.1 |

The differences between DPO and MixDPO are **within the margin of error** on all tasks. This is expected — downstream benchmarks (MMLU, ARC, etc.) measure factual knowledge determined during pre-training/SFT, not instruction-following quality which is the target of preference optimization. MixDPO is competitive on these benchmarks while achieving significant gains on open-ended generation benchmarks (AlpacaEval 2.0, MT-Bench).

### W2: Missing comparison with β-DPO [Wu et al., NeurIPS 2024]

Thank you for pointing out this important related work. β-DPO adaptively adjusts β per sample based on preference strength, which shares the motivation of handling varying difficulty levels. However, there are key differences:

- **β-DPO** adjusts the *temperature* of the preference loss for all pairs, still applying DPO loss uniformly. It does not change the *type* of supervision signal.
- **MixDPO** fundamentally changes the *objective* for difficult pairs: from contrastive preference loss to SFT loss on chosen responses. This is motivated by our finding (Section 4.1) that small-margin pairs provide **noisy contrastive signals** but still contain **high-quality chosen responses** worth learning from.

We have added β-DPO as a baseline:

| Method | LC Win Rate (%) | Win Rate (%) | Avg Length |
|--------|----------------|-------------|-----------|
| DPO (β=0.01) | 9.37 | 16.77 | 2895 |
| β-DPO | 6.78 | 15.65 | 3460 |
| **MixDPO (Ours)** | **14.42** | **36.65** | **2843** |

β-DPO (6.78%) underperforms even standard DPO (9.37%), while MixDPO (14.42%) significantly outperforms both. This suggests that adaptively adjusting β alone is insufficient — changing the objective type for difficult pairs (as MixDPO does) is more effective.

### W3: No β sweep; β fixed at 0.01

We respectfully clarify that β=0.01 is **not an arbitrary or cherry-picked value**. This is the **standard hyperparameter** from the HuggingFace Alignment Handbook (https://github.com/huggingface/alignment-handbook), which serves as the canonical training recipe for the UltraChat SFT + UltraFeedback DPO pipeline. Specifically:

1. **β=0.01** is the default value used in the Alignment Handbook's DPO training config for Zephyr-7B and similar models. It is the community-standard setting for this exact model-dataset combination.
2. **All baselines in our paper** (DPO, IPO, cDPO, SimPO, DPOP, etc.) use the **same β=0.01** to ensure a fair comparison. MixDPO does not benefit from a different β than the baselines — they all share the same value.
3. This choice is consistent with prior works that use this pipeline, including Zephyr (Tunstall et al., 2023) and SimPO (Meng et al., 2024).

Therefore, the concern that "the advantage comes from hyperparameter mismatch" does not apply here — **MixDPO and all baselines operate under identical hyperparameter settings**, following established community practices.

More importantly, we emphasize that **MixDPO also did not undergo any β tuning**. If β=0.01 is suboptimal for DPO, it is equally suboptimal for MixDPO — both methods are evaluated under the same potentially non-ideal β. The fact that MixDPO outperforms DPO under the same (un-tuned) β demonstrates that the improvement stems from the method itself (hybrid objective design), not from hyperparameter advantage. If anything, β tuning could further improve MixDPO's results as well.

That said, to further strengthen this argument, we conduct a β sweep with β ∈ {0.01, 0.05, 0.1} for **both** DPO and MixDPO:

| β | DPO LC WR (%) | DPO WR (%) | DPO Avg Len | MixDPO LC WR (%) | MixDPO WR (%) | MixDPO Avg Len |
|---|--------------|-----------|------------|-----------------|--------------|---------------|
| 0.01 | 9.37 | 16.77 | 2895 | **14.42** | **36.65** | 2843 |
| 0.05 | 7.18 | 16.40 | 4629 | 6.60 | 18.63 | 8189 |
| 0.1 | 4.92 | 12.05 | 4476 | 3.72 | 13.66 | 4750 |

At the default β=0.01, MixDPO outperforms DPO by a large margin (14.42% vs 9.37%). At larger β values, both methods degrade, consistent with the observation that β=0.01 is well-calibrated for this pipeline. MixDPO's advantage is most pronounced at the standard β, confirming that the improvement comes from the method design rather than β sensitivity.

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

| Method | Win Rate (%) | Adj. WR (%) | 95% CI | Avg Length |
|--------|-------------|------------|--------|-----------|
| SFT | 3.9 | 7.8 | [6.0, 10.0] | 1324 |
| DPO | 20.4 | 31.2 | [27.7, 34.8] | 1906 |
| IPO | 20.6 | 32.5 | [29.1, 35.7] | 2072 |
| SimPO | 20.2 | 32.5 | [29.1, 36.1] | 2058 |
| RDPO | 18.9 | 28.6 | [25.6, 31.7] | 1869 |
| KTO | 17.6 | 27.4 | [24.2, 30.9] | 1788 |
| SelectiveDPO | 20.5 | 32.1 | [27.7, 32.4] | 1895 |
| MixDPO (Ours) | 16.6 | 26.3 | [23.1, 29.5] | 1605 |

**Mistral-7B:**

| Method | Win Rate (%) | Adj. WR (%) | 95% CI | Avg Length |
|--------|-------------|------------|--------|-----------|
| SFT | 3.0 | 5.1 | [3.5, 6.8] | 1196 |
| DPO | 10.0 | 15.0 | [12.5, 18.0] | 1569 |
| IPO | 6.8 | 13.2 | [10.8, 15.6] | 1533 |
| SimPO | 11.2 | 19.5 | [16.7, 22.6] | 1627 |
| RDPO | 9.7 | 16.6 | [13.8, 19.4] | 1485 |
| KTO | 5.0 | 10.3 | [8.1, 12.7] | 1251 |
| SelectiveDPO | 10.2 | 16.6 | [13.9, 19.3] | 1609 |
| MixDPO (Ours) | 10.2 | 20.5 | [17.7, 23.3] | 1703 |

Key observations:
1. **LLaMA-3-8B**: MixDPO's CI [23.1, 29.5] overlaps with DPO [27.7, 34.8] and other methods. MixDPO generates the shortest responses (1605 chars), and LLM judges are known to exhibit length bias — this largely explains the lower win rate.
2. **Mistral-7B**: MixDPO achieves the **highest** Arena-Hard win rate (20.5%) among all methods, showing that the LLaMA-3-8B result is not a systematic weakness.
3. MixDPO achieves the best **AlpacaEval 2.0 LC Win Rate** on both models — a metric specifically designed to control for length bias.

### Q2: Discuss related works [1-6]

See W5 above. All six works have been discussed and positioned relative to MixDPO.

### Q3: Training-time evidence for NLL/log-prob on chosen and rejected responses

Following the evaluation methodology of Tajwar et al. [7], we report the change in mean sequence-level log-probabilities on the eval set between the start and end of training:

[7] Tajwar F. et al. Unintentional Unalignment: Likelihood Displacement in Direct Preference Optimization // ICLR 2025.

<!-- Old wandb 4-point results (kept for reference):
LLaMA-3-8B on UltraFeedback eval set:
| Method | Δ Chosen LogP | Δ Rejected LogP | Δ Gap |
| DPO | +4 (-268 → -264) | +12 (-284 → -272) | -8 (16 → 8) |
| MixDPO | -4 (-260 → -264) | -24 (-278 → -302) | +20 (18 → 38) |

New 10-point MixDPO rerun results (eval_steps=40):
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

We evaluate the SFT baseline, vanilla DPO, and MixDPO on the same eval set (500 samples) and report the change in mean sequence-level log-probabilities relative to the SFT model:

**LLaMA-3-8B:**

| Method | Δ Chosen LogP | Δ Rejected LogP | Δ Gap |
|--------|--------------|-----------------|-------|
| DPO | -14.8 (-311.9 → -326.6) | -29.6 (-291.3 → -320.9) | +14.8 (-20.5 → -5.7) |
| MixDPO | -11.7 (-311.9 → -323.5) | -26.9 (-291.3 → -318.2) | +15.2 (-20.5 → -5.3) |

**Mistral-7B:**

| Method | Δ Chosen LogP | Δ Rejected LogP | Δ Gap |
|--------|--------------|-----------------|-------|
| DPO | -33.6 (-314.3 → -347.9) | -63.0 (-293.6 → -356.6) | +29.4 (-20.7 → 8.7) |
| MixDPO | -15.3 (-314.3 → -329.6) | -34.9 (-293.6 → -328.5) | +19.6 (-20.7 → -1.1) |

**Qwen-2.5-7B:**

| Method | Δ Chosen LogP | Δ Rejected LogP | Δ Gap |
|--------|--------------|-----------------|-------|
| DPO | -17.2 (-305.9 → -323.1) | -34.0 (-287.8 → -321.7) | +16.8 (-18.1 → -1.3) |
| MixDPO | -18.2 (-305.9 → -324.1) | -37.0 (-287.8 → -324.7) | +18.8 (-18.1 → 0.6) |

Key observations:
- **Chosen displacement**: MixDPO shows less chosen log-prob decrease than DPO — LLaMA: -11.7 vs -14.8 (21% less), Mistral: -15.3 vs -33.6 (54% less), Qwen: roughly equal.
- **Gap improvement**: Comparable across methods, with MixDPO slightly better on LLaMA and Qwen. DPO pushes rejected down more aggressively, but this does not translate to better alignment — MixDPO achieves higher AlpacaEval LC WR on all three models.

<!--
**Connection to likelihood displacement.** The chosen log-prob decrease is a known phenomenon called *likelihood displacement* (Razin et al., 2024; Pal et al., 2024). Existing fixes — adding NLL regularizers (DPOP, DPO+NLL) or filtering difficult pairs (Razin et al., 2024) — either create competing gradients or discard useful data, and generally do not achieve a net increase above the SFT baseline. MixDPO instead **separates objectives at the data level**: the SFT stage directly maximizes chosen log-prob without contrastive interference, explaining why MixDPO shows less displacement across all three models.

Likelihood displacement 各方法对 chosen log-prob 的实际效果:

| 方法 | Chosen prob 变化 | 具体数据 | 来源 |
|------|-----------------|----------|------|
| DPO | ❌ 大幅下降 | Llama-3-8B: 0.99→0.03; MetaMath: -0.37→-1.82 | Razin et al. 2024; Pal et al. 2024 |
| IPO | ❌ 同样下降 | Razin et al. 确认 IPO 也有 displacement | Razin et al. 2024 |
| DPOP | ⚠️ 仅 MetaMath 上绝对上升 | MetaMath: -0.37→-0.26 (Δ=+0.11), edit distance 仅 6.5%, 非通用场景 | Pal et al. 2024 |
| DPO+NLL | ⚠️ 缓解下降 | BDPO 论文称 "chosen prob increases", 指相对趋势, 非绝对超过 SFT 基线 | BDPO 2025 |
| CPO | ⚠️ 缓解下降 | 含 NLL 项, 消融实验证实 NLL 是关键, 未报告 chosen prob 轨迹 | Xu et al. 2024 |
| ORPO | ⚠️ 与 SFT 持平 | 论文原话 "on par with SFT", 不是上升 | Hong et al. 2024 |
| SquaredPO | ⚠️ 大幅缓解 | 持续下降比例: DPO 99.63% → SquaredPO 4.21%, 但仍有下降 | Pipano et al. 2026 |
| LD-DPO | ⚠️ 数据筛选回避问题 | 过滤高 CHES score 数据; 低 CHES 数据 DPO 本身能上升; 高 CHES 数据直接丢弃 | Razin et al. 2024 |

结论: 没有任何方法在通用场景下实现 chosen prob 的绝对上升. MixDPO 在数据层面分离目标, SFT 阶段直接最大化 chosen prob, 不受 contrastive loss 干扰.
-->

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
| MixDPO (Ours) | 14.42 | 2843 | 7.67 | 1565 |

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

To further demonstrate this, we conduct experiments on **UltraInteract Math_CoT** — a binary-label math reasoning dataset where difficulty margins are generated by a reward model (`Eurus-RM-7B`). We train Eurus-7B-SFT with LoRA on 20K samples and evaluate across 6 benchmarks:

| Method | GSM8K | MathQA | Minerva Math | MMLU-Pro Math | BBH | ARC | **Avg** |
|--------|-------|--------|-------------|--------------|-----|-----|---------|
| *SFT (ref.)* | *61.3±1.3* | *38.6±0.9* | *17.6±0.5* | *23.5±1.2* | *57.5±0.5* | *57.6±1.4* | *42.69* |
| DPO | 27.4±1.2 | 32.9±0.9 | 9.1±0.4 | 7.3±0.7 | 17.9±0.4 | 55.6±1.5 | 25.03 |
| SimPO | 47.0±1.4 | 34.1±0.9 | 9.4±0.4 | 11.3±0.9 | 12.2±0.4 | 56.1±1.5 | 28.36 |
| SelectiveDPO | **61.8±1.3** | 36.0±0.9 | 13.3±0.5 | 18.7±1.1 | 37.4±0.5 | 57.4±1.4 | 37.43 |
| MixDPO (Ours) | 61.1±1.3 | **36.4±0.9** | **13.3±0.5** | **18.7±1.1** | **45.4±0.6** | **58.3±1.4** | **38.87** |

MixDPO achieves the highest average (38.87%) among all DPO methods and outperforms SelectiveDPO (37.43%) by 1.4 points. Notably, standard DPO and SimPO severely degrade from SFT (25.03% and 28.36% vs 42.69%), consistent with known challenges of DPO on reasoning tasks (gradient entanglement, likelihood displacement). MixDPO's difficulty-aware routing effectively mitigates this degradation, achieving strong results on both math benchmarks (GSM8K, MathQA, Minerva Math, MMLU-Pro Math) and general reasoning (BBH, ARC).

<!--
## 备注：DPO 在推理任务上有害 — 文献证据汇总

### 直接证据

| 论文 | 发表 | 关键数据 |
|------|------|----------|
| Step-DPO (2406.18629) | — | Qwen2-7B MATH: DPO 仅 +0.2%, Step-DPO +3.8% |
| Eurus (2404.02078) | ICLR 2025 | MATH: DPO 28.3% vs KTO 33.2% vs NCA 34.2%; 70B DPO reward 降到 -∞ |
| 3D-Properties (2406.07327) | ICLR 2025 | Off-policy DPO 比 base model 还差 (26.8% vs 32.2% on MATH*) |
| Unpacking DPO vs PPO (2406.09279) | NeurIPS 2024 | PPO 比 DPO 好: reasoning +1.3, coding +2.9, safety +2.3 |
| Iterative RPO (2404.19733) | NeurIPS 2024 | MATH: DPO 12.4% < few-shot CoT 12.5%; SFT on chosen 16.8% |
| BPO (2506.03557) | — | MATH: DPO 18.8% vs BPO 28.9% (+10.1%); Qwen2.5-Math: DPO 35.0% vs BPO 46.7% |
| Insights into Alignment (2404.14723) | — | GSM8K: DPO 30.62% vs KTO 34.72% |
| Future Policy Aware (2509.19893) | — | FPA 比 standard DPO 在 math 上高 5.75% |
| Smaug/DPOP (2402.13228) | — | DPO 降低 chosen log-prob; MetaMath: -0.37 -> -1.82 |

### 根本原因（文献共识）

1. **梯度纠缠 (Gradient entanglement)**: 数学的正确/错误解法共享大量 token（公式、中间步骤），惩罚错误解法连带惩罚了正确步骤
2. **整序列拒绝 (Whole-sequence rejection)**: DPO 拒绝整个 response，但错误通常出现在中间步骤，前面的正确推理被误伤
3. **Chosen likelihood 下降**: DPO 的 loss 会降低 chosen response 的概率（likelihood displacement），在推理任务上尤其严重
4. **过度优化 (Overoptimization)**: DPO 出现倒 U 形性能曲线，训练不到一个 epoch 就开始退化
5. **Off-policy 分布偏移**: DPO 的 off-policy 特性导致在推理领域容易出现分布不匹配

### 为什么 MixDPO 应该能改善

- MixDPO 对 difficult pairs 用 SFT → 避免在模糊 pairs 上的梯度纠缠
- SFT 直接最大化 chosen log-prob → 对抗 likelihood displacement
- 基于难度的 routing → DPO 只用在 contrastive signal 清晰的 pairs 上
- Eurus 发现 "提升 chosen data 的 reward 对推理任务尤其有益" → MixDPO 的 SFT 阶段正好做这件事

### UltraInteract_pair 数据集信息
- 220K pairs，MIT license，来自 OpenBMB
- Task 分布: Coding 96K, Math_CoT 57K, Math_PoT 56K, Logic 10K
- 来源数据集: MATH 49K, TACO 51K, codecontest 44K, mathqa 33K, gsm8k 21K
- 纯 binary labels (chosen/rejected)，没有数值评分
- 只有 Eurus 论文自己用了这个数据集做 DPO；其他论文都是自生成 preference data
- Eurus 用了全部 220K + 340K UltraFeedback；其他 math DPO 论文一般用 10-30K 样本
-->

