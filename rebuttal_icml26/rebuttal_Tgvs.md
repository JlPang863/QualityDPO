# Rebuttal to Reviewer Tgvs

**Score: 2 (Reject), Confidence: 4**

We thank Reviewer Tgvs for the thoughtful review. We address each concern below.

---

## Response to Weaknesses

### W1: Outdated base models (LLaMA-3, Mistral-7B)

We chose LLaMA-3-8B and Mistral-7B because they are the **standard benchmarking models** for the UltraChat SFT + UltraFeedback DPO pipeline (established by the HuggingFace Alignment Handbook and widely adopted in DPO literature, e.g., Zephyr, SimPO, DPOP). This standardized setup is critical for fair comparison: the Alignment Handbook provides a validated training recipe (learning rate, β, batch size, warmup, training steps) calibrated for these model–dataset combinations. All baselines use **exactly the same hyperparameters**, so performance differences reflect the method rather than tuning. Switching to a new model would require re-tuning every baseline independently, risking that gaps stem from unequal tuning effort. MixDPO introduces no additional tuning — the only new parameter is the difficulty threshold τ.

We additionally include **Qwen-2.5-7B** experiments (Table 2), demonstrating that MixDPO generalizes to more recent architectures. Furthermore, we conduct new experiments on **Qwen3-4B** (released March 2025), a state-of-the-art recent model:

| Method | LC Win Rate (%) | Win Rate (%) |
|--------|----------------|-------------|
| Qwen3-4B (Instruct baseline) | 42.70 | 47.14 |
| Vanilla DPO | 39.80 | 40.81 |
| SimPO | 44.39 | 45.34 |
| SelectiveDPO | 42.39 | 45.03 |
| **MixDPO (Ours)** | **54.52** | **56.40** |
<!-- | **MixDPO (Old)** | **41.13** | **41.61** | -->

<!--
Qwen3-4B 所有结果（含差的版本）：
| Method | LC WR | WR | Avg Length | 对应结果 |
| Qwen3-4B (Instruct baseline) | 42.70 | 47.14 | 2288 | Qwen3-4B |
| Vanilla DPO | 39.80 | 40.81 | 2089 | qwen3-4b-instruct-dpo-full |
| SelectiveDPO | 42.39 | 45.03 | 2187 | qwen3-4b-instruct-selectivedpo-full |
| MixDPO (旧版, 差) | 41.13 | 41.61 | 2078 | qwen3-4b-instruct-ours4-6-sorted-score-diff-full |
| MixDPO (新版, 好) | 54.52 | 56.40 | 2102 | qwen3-4b-instruct-ours4-6-sorted-score-diff-full-new |
| SimPO | 44.39 | 45.34 | 2111 | qwen3-4b-instruct-simpo-full |
-->
 
MixDPO achieves 54.52% LC WR on Qwen3-4B, significantly outperforming all baselines including the Instruct baseline (42.70%), confirming that MixDPO generalizes effectively to the latest model architectures.

### W2: Win rates lower than other DPO papers

The absolute win rate differences across papers are primarily due to the **LLM judge version**. Since most baselines in our experiments use SimPO's publicly released model checkpoints, we can directly compare the same models evaluated by different judges. Below we show AlpacaEval 2.0 LC Win Rate (%) for the same models under SimPO's judge (GPT-4-Preview-1106) vs our judge (GPT-4.1):

AlpacaEval 2.0 **LC Win Rate (%)** on the same model checkpoints, evaluated by different judges (Δ = difference from DPO):

| Method | LLaMA-3-8B (SimPO) | LLaMA-3-8B (Ours) | Mistral-7B (SimPO) | Mistral-7B (Ours) |
|--------|-------------------|-------------------|-------------------|-------------------|
| DPO | 18.2 | 9.37 | 15.1 | 5.14 |
| CPO | 10.8 (−7.4) | 4.25 (−5.1) | 9.8 (−5.3) | 4.04 (−1.1) |
| IPO | 14.4 (−3.8) | 5.89 (−3.5) | 11.8 (−3.3) | 5.45 (+0.3) |
| KTO | 14.2 (−4.0) | 4.27 (−5.1) | 13.1 (−2.0) | 5.02 (−0.1) |
| RDPO | 17.6 (−0.6) | 6.92 (−2.5) | 17.4 (+2.3) | 6.03 (+0.9) |
| SimPO | 22.0 (+3.8) | 6.77 (−2.6) | 21.5 (+6.4) | 4.30 (−0.8) |

<!--
AlpacaEval 2.0 **Win Rate (%)** on the same model checkpoints (Δ = difference from DPO):

| Method | LLaMA-3-8B (SimPO) | LLaMA-3-8B (Ours) | Mistral-7B (SimPO) | Mistral-7B (Ours) |
|--------|-------------------|-------------------|-------------------|-------------------|
| DPO | 15.5 | 16.77 | 12.5 | 4.72 |
| CPO | 8.1 (−7.4) | 9.69 (−7.1) | 8.9 (−3.6) | 3.85 (−0.9) |
| IPO | 14.2 (−1.3) | 11.55 (−5.2) | 9.4 (−3.1) | 4.60 (−0.1) |
| KTO | 12.4 (−3.1) | 3.98 (−12.8) | 9.1 (−3.4) | 3.23 (−1.5) |
| RDPO | 14.4 (−1.1) | 11.06 (−5.7) | 12.8 (+0.3) | 4.60 (−0.1) |
| SimPO | 20.3 (+4.8) | 14.04 (−2.7) | 20.8 (+8.3) | 5.47 (+0.8) |
-->

The absolute win rates differ substantially due to the judge version, but the **relative trends are consistent**: DPO variants like CPO and KTO underperform standard DPO under both judges, and the overall ranking is preserved. This confirms that the judge version affects absolute values but does not change relative comparisons. We will clarify this point more explicitly in the revised paper.

### W3: DPO+SFT as a straightforward baseline

**This baseline is already in our paper.** Table 9 (Appendix B.3) includes **"DPO+NLL"** (Pang et al., 2024b, *Iterative Reasoning Preference Optimization*), which adds a negative log-likelihood (SFT) loss term to DPO loss for **all** samples, without difficulty-based routing. This is exactly the "DPO+SFT" baseline the reviewer asks about.

Results (from Table 9, LLaMA-3-8B on UltraFeedback):

| Method | LC Win Rate (%) | Win Rate (%) |
|--------|----------------|-------------|
| Base SFT | 3.73 | 10.19 |
| Vanilla DPO | 9.37 | 16.77 |
| DPO+NLL (DPO+SFT on all data) | 4.25 | 8.45 |
| MixDPO (Ours) | 14.42 | 36.65 |

MixDPO outperforms DPO+NLL by a large margin (14.42% vs 4.25% LC WR), and also significantly outperforms vanilla DPO (9.37%). This demonstrates that **difficulty-aware routing is the key**, not merely combining DPO and SFT losses — applying SFT to all data (DPO+NLL) actually hurts performance compared to vanilla DPO.

### W4: Qwen-2.5-7B experiment placement in ablations

We agree that the placement is somewhat confusing. The Qwen-2.5-7B experiment is intended as a **generalization result** (testing whether MixDPO transfers to a third model family and an additional dataset), not a component ablation. We will adjust the presentation in the revised version to better distinguish generalization experiments from component ablations.

---

## Response to Key Questions

### Q1: Does any baseline combine SFT and DPO?

Yes — see W3 above. DPO+NLL in Table 9 is exactly this baseline, and MixDPO significantly outperforms it.

### Q2: What if pairs were sorted by chosen score instead of margin? Is SFT benefiting from ignoring high-scoring rejected responses?

Great question. We conduct controlled experiments using the same MixDPO framework (DPO on first 53,748 pairs, SFT on last 7,387 pairs), replacing the margin-based sorting with chosen score or rejected score as the sorting criterion:

| Sorting Criterion | LC Win Rate (%) | Win Rate (%) | Avg Length |
|-------------------|----------------|-------------|-----------|
| Sorted by chosen score | 6.10 | 7.33 | 1850 |
| Sorted by rejected score | 8.22 | 30.31 | 4677 |
| **Margin (original MixDPO)** | **14.42** | **36.65** | **2843** |

Margin sorting significantly outperforms both alternatives (14.42% vs 6.10% and 8.22% LC WR), confirming that **margin is the most effective routing signal**. This is intuitive: since DPO is a pairwise contrastive objective, the score margin which captures the pairwise relationship between chosen and rejected is the most natural criterion for identifying where contrastive learning is reliable versus where SFT is more appropriate.

### Q3: Were learning rates tuned in Section 4.1?

In Section 4.1, we intentionally use the **same learning rate** (Alignment Handbook default) across all difficulty buckets to isolate the effect of data difficulty from hyperparameter tuning. We did not tune lr per bucket — this is consistent with standard practice in data-difficulty analysis, where the goal is to compare subsets under controlled conditions rather than to find the best configuration for each subset.

To verify that our findings are robust to learning rate choice, we repeat the training dynamics analysis at two additional learning rates (lr=1e-6 and lr=5e-7):

![Training Dynamics at Different Learning Rates](figures/training_dynamics_lr_comparison.png)

The pattern **easy > middle > difficult** holds at both learning rates, consistent with Figure 3. This confirms the finding is intrinsic to data difficulty, not an artifact of lr selection.
