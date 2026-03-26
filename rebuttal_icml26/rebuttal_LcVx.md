# Rebuttal to Reviewer LcVx

**Score: 4 (Weak Accept), Confidence: 4**

We sincerely thank Reviewer LcVx for the positive assessment and constructive feedback.

---

## Response to Weaknesses

### W1: Rely on noisy LLM rating data — MixDPO seems vulnerable to noise

We agree that robustness to noisy scores is important. Appendix B.6 shows that under a 20% mislabeled setting (swapping the last 10% easy pairs with the most difficult ones), MixDPO remains comparable to baselines — it does not collapse under moderate score corruption, though performance does depend on rating reliability. We will position robustness to noisy ratings as an important future direction and discuss practical mitigations (e.g., preprocessing raw scores before computing difficulty).

### W2: "No additional computational overhead" claim is too strong

We agree and will revise. A more accurate statement: MixDPO introduces **no additional overhead when rating scores are already available** in the dataset. For datasets with only binary labels (e.g., our SimPO experiments where we used GPT-4o-mini to generate ratings), additional cost is indeed required.

### W3: Typo on Figure 1 label

Thank you for pointing this out. We apologize for this confusion. In AlpacaEval 2.0, "GPT-4-Turbo" on the y-axis refers to the **reference responses** that model outputs are compared against, while the **LLM judge** (GPT-4.1) is a separate role. We will revise the caption to make this distinction clearer.

### W4: No code released

Thank you for the note. We appreciate the reviewer's recognition that the appendix already provides substantial details about the training setup. We agree that releasing code would further improve reproducibility, and we plan to clean up and release the code and scripts to make the empirical setup easier to verify and build upon.

---

## Response to Key Questions

### Q1: Soft-weighting schedule — continuous weight instead of binary switch?

We intentionally use a binary z to isolate the effect of objective reassignment, without adding design choices from a weighting schedule. A soft transition may better capture the gradual nature of difficulty, but introduces questions about weighting form and optimization stability. We will discuss this as a promising future direction.

### Q2: Can MixDPO work with binary preferences only?

In the current formulation of MixDPO, we do require a **difficulty signal** to distinguish easy from difficult preference pairs. Therefore, if a dataset only provides binary chosen/rejected labels and no additional scalar signal, then binary preference alone is not sufficient to directly instantiate the current version of MixDPO. In that regime, an additional difficulty-estimation step would indeed be needed, and we agree that this introduces overhead. We will discuss this limitation in the revised version.