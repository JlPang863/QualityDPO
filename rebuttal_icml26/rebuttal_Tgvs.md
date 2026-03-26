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

Results:
| Method | LC Win Rate (%) |
|--------|----------------|
| DPO+NLL (DPO+SFT on all data) | 4.25 |
| MixDPO | **14.42** |

MixDPO outperforms DPO+NLL by a large margin (14.42% vs 4.25%), demonstrating that **difficulty-aware routing is the key**, not merely combining DPO and SFT losses.

### W4: Qwen-2.5-7B experiment placement in ablations

Thank you for this presentation-related comment. The Qwen-2.5-7B experiment is intended as a **generalization result on an additional base model**, rather than as a standard component ablation. Our main benchmark table (Table 1) focuses on two representative base models with a broader set of baselines, while Table 2 is used to test whether the method also transfers to a third model family (Qwen-2.5-7B) and to an additional preference dataset (Argilla-7k). This is why the Qwen experiment currently appears in Section 6.1 together with the dataset-generalization result. We agree, however, that the current organization under the broader "ablation" section can make this less clear. We will revise the presentation to better distinguish **generalization experiments** from component ablations, so that the role of the Qwen-2.5-7B result is more immediately clear to the reader.

---

## Response to Key Questions

### Q1: Does any baseline combine SFT and DPO?

Yes — see W3 above. DPO+NLL in Table 9 is exactly this baseline, and MixDPO significantly outperforms it.

### Q2: What if pairs were sorted by chosen score instead of margin? Is SFT benefiting from ignoring high-scoring rejected responses?

This is an insightful question. We have conducted an additional experiment: **MixDPO with data sorted by chosen score** (descending, high-quality chosen first) instead of by margin.

> **TODO: Add results from `mixdpo-sorted-chosen-score` experiment.**

This experiment disentangles two hypotheses:
- **(A) Margin hypothesis**: MixDPO works because margin-based difficulty correctly identifies pairs where DPO loss is unreliable.
- **(B) Chosen-quality hypothesis**: MixDPO works simply because high-scoring chosen responses are easy to learn from, regardless of the rejected response.

If margin sorting outperforms chosen-score sorting, it confirms hypothesis (A) — the margin signal provides information beyond chosen response quality alone.

Additionally, we note that the SFT phase in MixDPO is applied to **chosen responses from difficult pairs** (low margin). These chosen responses have similar quality to the rejected responses (by definition of low margin), so the SFT phase is not simply "ignoring high-scoring negatives" — it is extracting useful supervision from ambiguous pairs where the preference signal is too noisy for contrastive learning.

### Q3: Were learning rates tuned in Section 4.1?

In Section 4.1, we use the **same learning rate** (from the Alignment Handbook default) across all difficulty buckets to isolate the effect of data difficulty from hyperparameter tuning. We acknowledge that different buckets may benefit from different learning rates. However, our main finding — that difficult pairs harm DPO but help SFT — holds across the learning rate sweep we report in the appendix (Table 7, with lr ∈ {5e-7, 8e-7, 1e-6}), where MixDPO consistently outperforms baselines at each learning rate.
