# Rebuttal to Reviewer Tgvs

**Score: 2 (Reject), Confidence: 4**

We thank Reviewer Tgvs for the thoughtful review. We address each concern below.

---

## Response to Weaknesses

### W1: Outdated base models (LLaMA-3, Mistral-7B)

We chose LLaMA-3-8B and Mistral-7B because they are the **standard benchmarking models** for the UltraChat SFT + UltraFeedback DPO pipeline (established by the HuggingFace Alignment Handbook and widely adopted in DPO literature, e.g., Zephyr, SimPO, DPOP). This standardized setup is critical for fair comparison: the Alignment Handbook provides a validated training recipe (learning rate, β, batch size, warmup, training steps) calibrated for these model–dataset combinations. All baselines use **exactly the same hyperparameters**, so performance differences reflect the method rather than tuning. Switching to a new model would require re-tuning every baseline independently, risking that gaps stem from unequal tuning effort. MixDPO introduces no additional tuning — the only new parameter is the difficulty threshold τ.

We additionally include **Qwen-2.5-7B** experiments (Table 2), demonstrating that MixDPO generalizes to more recent architectures.

### W2: Win rates lower than other DPO papers

The absolute win rate differences across papers are primarily due to the **LLM judge version**. We use GPT-4-1106 (GPT-4-Turbo) as the AlpacaEval 2.0 judge for cost efficiency, whereas some recent papers use GPT-4o or newer versions, which tend to assign higher absolute win rates. This affects absolute numbers but not relative rankings.

Importantly, the **relative trends are consistent** with SimPO. As we discuss in Lines 323–328 of our paper, many DPO variants (e.g., CPO, KTO) fail to outperform standard DPO and in some cases underperform it — the same pattern observed by SimPO, which used GPT-4-Preview-1106 as the judge. Since most baselines in our experiments use SimPO's publicly released model checkpoints, the only difference is the LLM judge version. The fact that the relative ranking is preserved across different judges confirms that the judge version may affect absolute win rates but does not change relative comparisons. We will clarify this point more explicitly in the revised paper.

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

This is an insightful question. To disentangle the source of MixDPO's improvement, we conduct three controlled experiments using the same MixDPO framework (DPO on first 53,748 pairs, SFT on last 7,387 pairs), varying only the **sorting criterion** that determines which pairs end up in the SFT stage:

| Sorting Criterion | What the last 7,387 pairs are | Hypothesis tested |
|-------------------|-------------------------------|-------------------|
| Margin (original MixDPO) | Low margin (ambiguous) pairs | Margin-based difficulty is the key signal |
| Chosen score | Low chosen-score pairs | SFT benefits from chosen quality, not margin |
| Rejected score | Low rejected-score pairs | SFT benefits from ignoring high-scoring negatives |

> **TODO: Add AlpacaEval results for sorted-chosen-score and sorted-rejected-score.**

If margin sorting outperforms the other two, it confirms that **the margin signal provides information beyond chosen or rejected quality alone** — the pairwise relationship matters.

Additionally, we note that in low-margin pairs, chosen and rejected responses have similar quality by definition. The SFT phase is therefore not simply "ignoring high-scoring negatives" — it is extracting useful supervision from pairs where the preference signal is too noisy for contrastive learning.

### Q3: Were learning rates tuned in Section 4.1?

In Section 4.1, we intentionally use the **same learning rate** (Alignment Handbook default) across all difficulty buckets to isolate the effect of data difficulty from hyperparameter tuning. We did not tune lr per bucket — this is consistent with standard practice in data-difficulty analysis, where the goal is to compare subsets under controlled conditions rather than to find the best configuration for each subset. We acknowledge that per-bucket lr tuning could potentially narrow the gap between easy and difficult subsets. We will add this as a caveat in the revision.
