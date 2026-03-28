# Rebuttal to Reviewer BC6i

**Score: 3 (Weak Reject), Confidence: 3**

We thank Reviewer BC6i for the constructive feedback. We address each concern below.

---

## Response to Weaknesses

### W1: Ablation incomplete — need "DPO on easy only" and "SFT on hard only" extreme controls

These extreme-case controls are already included in our paper:

- **"DPO on easy only"**: This is exactly the **"Ours + discard difficult"** variant in **Figure 4a** (discussed in Lines 375–376), which applies DPO only to easy pairs and discards all difficult pairs. MixDPO outperforms this variant, demonstrating that difficult pairs provide valuable supervision when trained with SFT.
- **"SFT on hard only"**: We have conducted additional experiments training SFT only on difficult pairs (score_diff < 0.5) using chosen, rejected, or both responses:

| SFT Target | # Samples | LC Win Rate (%) | Win Rate (%) | Avg Length |
|------------|-----------|----------------|-------------|------------|
| Chosen only | 7,387 | 2.11±0.14 | 1.93±0.48 | 1327 |
| Rejected + Chosen | 14,774 | 2.33±0.15 | 2.24±0.52 | 1394 |
| Rejected only | 7,387 | 2.91±0.19 | 2.86±0.59 | 1448 |

All three SFT-only settings perform poorly on AlpacaEval 2.0 compared to MixDPO (LC WR = 14.42%), confirming that pure SFT on difficult pairs alone is insufficient — the DPO phase on easy pairs is essential. We also provide additional Full SFT baselines in Appendix B.5 (Table 12), which further support this conclusion.

Together, these results confirm that MixDPO's hybrid design (DPO on easy + SFT on hard) outperforms both extremes.

### W2: Arena-Hard inconsistency — MixDPO worse than DPO/IPO/SimPO/SelectiveDPO on LLaMA-3-8B

We provide a detailed analysis. First, we note that this inconsistency is **specific to LLaMA-3-8B** — on Mistral-7B, MixDPO achieves the **highest** Arena-Hard win rate (20.5%) among all methods.

For LLaMA-3-8B, we have added 95% bootstrap CIs:

| Method | Win Rate (%) | Adj. WR (%) | 95% CI | Avg Length |
|--------|-------------|------------|--------|-----------|
| DPO | 20.4 | 31.2 | [27.7, 34.8] | 1906 |
| IPO | 20.6 | 32.5 | [29.1, 35.7] | 2072 |
| SimPO | 20.2 | 32.5 | [29.1, 36.1] | 2058 |
| SelectiveDPO | 20.5 | 32.1 | [27.7, 32.4] | 1895 |
| **MixDPO (Ours)** | **16.6** | **26.3** | **[23.1, 29.5]** | **1605** |

We attribute MixDPO's lower LLaMA-3-8B Arena-Hard score to two factors:

**1. High evaluation variance.** MixDPO's CI [23.1, 29.5] overlaps with DPO [27.7, 34.8] and other methods, suggesting the differences are within evaluation uncertainty.

**2. Length bias.** MixDPO generates the shortest responses (1605 chars) while higher-ranked methods generate longer outputs (IPO: 2072, SimPO: 2058, DPO: 1906). LLM judges are known to exhibit length bias, which penalizes MixDPO on Arena-Hard's raw win rates.

Importantly, this is not a systematic weakness. On the other two base models, MixDPO achieves the **highest** Arena-Hard win rate:

**Mistral-7B Arena-Hard:**

| Method | Win Rate (%) | Adj. WR (%) | 95% CI | Avg Length |
|--------|-------------|------------|--------|-----------|
| DPO | 10.0 | 15.0 | [12.5, 18.0] | 1569 |
| SimPO | 11.2 | 19.5 | [16.7, 22.6] | 1627 |
| SelectiveDPO | 10.2 | 16.6 | [13.9, 19.3] | 1609 |
| **MixDPO (Ours)** | **10.2** | **20.5** | **[17.7, 23.3]** | **1703** |

**Qwen-2.5-7B Arena-Hard:**

| Method | Win Rate (%) | Adj. WR (%) | 95% CI | Avg Length |
|--------|-------------|------------|--------|-----------|
| DPO | 24.6 | 35.6 | [32.3, 39.1] | 1990 |
| SimPO | 23.2 | 35.1 | [31.5, 38.8] | 1878 |
| SelectiveDPO | 28.2 | 39.2 | [35.4, 42.9] | 2052 |
| **MixDPO (Ours)** | **28.6** | **39.8** | **[36.0, 43.8]** | **2019** |

On AlpacaEval 2.0 LC Win Rate — which controls for length bias — MixDPO consistently outperforms all baselines across all three models (LLaMA: 14.42%, Mistral: 7.67%, Qwen: 3.45%).

### W3: LLM judge dependency — MixDPO's improvement marginal on downstream tasks (Table 14)

We agree that LLM-as-a-judge evaluation can be noisy. However, this noise is shared across all compared methods, since MixDPO and all baselines are evaluated under the same GPT-4.1 judge and protocol. We therefore believe the results are informative for relative comparison. LLM-as-a-judge is also a widely adopted evaluation protocol in recent preference-alignment literature, and we follow this common practice by evaluating on three established benchmarks (AlpacaEval 2, Arena-Hard, and MT-Bench).

Regarding Table 14: we agree it should be interpreted conservatively. DPO-style methods are primarily designed to improve alignment rather than downstream-task accuracy, so we include Table 14 mainly as a sanity check — showing that MixDPO remains competitive and does not harm general capability. The primary evidence for MixDPO is the alignment-oriented evaluation in Table 1. We will revise the wording to make this distinction clearer.

### W4: Threshold τ sensitivity — how to select optimal τ for different tasks

We agree that the difficult-pair threshold τ is an important hyperparameter, and that a fixed raw threshold may not transfer directly across domains. Our current sensitivity study already shows that performance changes meaningfully with τ: on UltraFeedback, τ=0.5 (approximately the top 10% most difficult pairs) performs best, while larger thresholds corresponding to 25% and 50% difficult subsets lead to weaker results. This suggests the key issue is not simply "using more hard pairs," but selecting a **moderate** difficult subset that balances informativeness and optimization stability.

For cross-domain use, we do **not** view a single absolute τ value as universal, since score scales can differ across datasets. A more practical strategy is to select τ by **percentile / difficult-pair ratio** on a small validation set (e.g., sweeping top 5% / 10% / 20% / 30% difficult pairs), rather than relying on a fixed raw score gap. This is also more robust when rating scales differ across domains or data sources. Based on our results, using a relatively small difficult subset (around top 10%) is a strong default choice, while overly large difficult subsets are more likely to hurt performance. We will add this discussion and make the threshold-selection guideline more explicit in the revision.

---

## Response to Key Questions

### Q1: Arena-Hard inconsistency — what is the intuition?

Please see W2 above for a detailed analysis with 95% bootstrap CIs and generation length statistics. In summary:

1. **High evaluation variance**: Arena-Hard's bootstrap CIs are wide (±3–4%). MixDPO's CI [23.1, 29.5] overlaps with DPO [27.7, 34.8] and other methods, suggesting the differences are within evaluation uncertainty.
2. **Length bias**: MixDPO generates the shortest responses (avg 1605 tokens vs 1900–2070 for other methods). LLM judges favor longer outputs, which disproportionately penalizes MixDPO on Arena-Hard.
3. **Benchmark characteristics**: Arena-Hard favors verbose, detailed responses, while MixDPO's strength lies in improving general helpfulness and instruction-following quality — better captured by AlpacaEval 2.0 and MT-Bench.

On AlpacaEval 2.0 LC Win Rate (controlling for length bias), MixDPO achieves **14.42%**, significantly outperforming all baselines (DPO: 9.37%, SimPO: 6.92%, IPO: 5.89%).

### Q2: Claim (2) — SFT as stabilizing stage is not new. What is the true contribution?

We agree that the main contribution is **not** simply "using SFT as a stabilizing stage" — that idea overlaps with existing practice. We will revise Lines 86–88 to make this more precise. The true conceptual contribution has three parts:

1. **A data-centric observation about pair difficulty and objective choice.** Our key finding is that the usefulness of a preference pair depends strongly on the optimization objective. Figure 3 shows that low-margin pairs lead to degraded training dynamics under DPO — they are not simply unhelpful, but **mismatched with pairwise preference optimization**.

2. **Difficulty-conditioned objective reassignment, not generic SFT regularization.** Unlike ORPO or DPO+NLL which apply SFT as a general regularization term to all data (DPO+NLL achieves only 4.25% LC WR, Table 9), MixDPO uses pair difficulty to decide **which objective should be applied to which subset**: easy pairs → DPO, difficult pairs → chosen-only SFT. The novelty lies in **data-dependent routing of supervision**, rather than in SFT itself.

3. **Retaining hard pairs instead of discarding them.** Existing approaches (e.g., SelectiveDPO) treat low-margin pairs as noisy and filter them out. We challenge this view by showing that such pairs still provide useful supervision when matched with a more suitable objective. This "retain but reassign" perspective is the main conceptual distinction.

### Q3: Claim (1) — the observation about easy-to-difficult degradation is not new

We appreciate this point. Claim (1) (Lines 82–86) is intended as an **empirical finding** that motivates our method, not as a standalone novelty claim. What we view as novel is the specificity of this finding: using **rating score margin** as a difficulty metric, Figure 3 systematically shows that as margin decreases, DPO exhibits degraded training dynamics across multiple dimensions (convergence, reward accuracy, reward margins). Critically, the same difficult pairs become **beneficial under SFT** — this is the key insight that motivates MixDPO.

As evidence: pure curriculum DPO without objective switching ("DPO + sorted data" in Figure 4a) achieves 10.41% LC WR — only a modest gain over vanilla DPO (9.37%). In contrast, MixDPO achieves 14.42% by adding objective reassignment on difficult pairs. The major improvement (10.41→14.42%) comes from the routing, not the ordering. This confirms that the contribution is not curriculum learning itself, but the difficulty-conditioned objective reassignment it enables (see Q2).

### Q4: Why only use chosen (yw) for SFT? Test with rejected (yl) or both?

Our rationale is that, for low-margin pairs, the preferred response yw still provides the cleanest positive supervision signal, since it is by definition better than yl. If both yw and yl are used for SFT, the model is asked to imitate two different answers for the same prompt, even though one is known to be inferior. This can blur the supervision signal and partially cancel the benefit of using SFT as a stable fallback objective.

To validate this, we conduct additional experiments within the full MixDPO pipeline, replacing only the SFT target for difficult pairs while keeping the DPO phase on easy pairs identical (all resuming from the same checkpoint):

| SFT Target | LC Win Rate (%) | Win Rate (%) |
|------------|----------------|-------------|
| Rejected only | 6.29 | 24.60 |
| Rejected + Chosen | 6.64 | 25.47 |
| **Chosen only (MixDPO)** | **14.42** | **36.65** |

SFT on chosen only (MixDPO) significantly outperforms both alternatives, confirming our design choice. Using rejected responses — either alone or combined — provides some improvement over discarding difficult pairs entirely, but is far less effective than using chosen responses.

### Q5: What data is used for Base SFT model? What if SFT is trained on UltraFeedback yw?

The Base SFT models are publicly available models finetuned on UltraChat-200k (see Table 4 in Appendix A.2):
- LLaMA-3-8B-SFT: `princeton-nlp/Llama-3-Base-8B-SFT` (sourced from SimPO)
- Mistral-7B-SFT: `HuggingFaceH4/mistral-7b-sft-beta` (sourced from Alignment-Handbook repository) 

These follow the standard Alignment Handbook pipeline (SFT on UltraChat → DPO on UltraFeedback).

Regarding SFT on UltraFeedback yw: Appendix B.5 (Table 12) reports two SFT-style baselines trained on UltraFeedback: (1) training only on positive (chosen) responses, and (2) training on both positive and negative responses. Both settings perform poorly — SFT on positive responses fails to capture the preference structure necessary for alignment, and training on both responses yields similarly poor results.
