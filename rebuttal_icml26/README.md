# ICML 2026 Rebuttal - Submission 8747

## Directory Structure

```
rebuttal_icml26/
├── README.md                          # This file
├── paper.pdf                          # Submitted paper
├── reviewer_comments/
│   └── openreview_reviews.pdf         # All reviewer comments
├── rebuttal_Z34t.md                   # Rebuttal for Reviewer Z34t (Score: 2, Reject)
├── rebuttal_BC6i.md                   # Rebuttal for Reviewer BC6i (Score: 3, Weak Reject)
├── rebuttal_Tgvs.md                   # Rebuttal for Reviewer Tgvs (Score: 2, Reject)
├── rebuttal_LcVx.md                   # Rebuttal for Reviewer LcVx (Score: 4, Weak Accept)
└── figures/
    ├── likelihood_displacement_eval.png
    └── likelihood_displacement_eval.pdf
```

## Reviewer Scores Summary

| Reviewer | Score | Recommendation | Confidence |
|----------|-------|----------------|------------|
| BC6i     | 3     | Weak Reject    | 3          |
| LcVx     | 4     | Weak Accept    | 4          |
| Z34t     | 2     | Reject         | 4          |
| Tgvs     | 2     | Reject         | 4          |

## Rebuttal Experiments

Training script: `run_all_cl_icml26_rebuttal.sh`

| # | Experiment | Reviewer | yaml |
|---|-----------|----------|------|
| 1 | beta-DPO baseline | Z34t W2 | `llama-3-8b-base-beta-dpo.yaml` |
| 2 | DPO beta=0.05 | Z34t W3 | `llama-3-8b-base-dpo-beta005.yaml` |
| 3 | DPO beta=0.1 | Z34t W3 | `llama-3-8b-base-dpo-beta01.yaml` |
| 4 | MixDPO beta=0.05 | Z34t W3 | `llama-3-8b-base-mixdpo-beta005.yaml` |
| 5 | MixDPO beta=0.1 | Z34t W3 | `llama-3-8b-base-mixdpo-beta01.yaml` |
| 6 | MixDPO sorted by chosen score | Tgvs Q2 | `llama-3-8b-base-mixdpo-sorted-chosen-score.yaml` |
| 7 | MixDPO rerun (checkpoint-336) | Z34t Q3 | `llama-3-8b-base-ours4-6-sorted-score-diff-full-rerun.yaml` |
| 8 | SFT on rejected (resume) | BC6i Q4 | `llama-3-8b-base-ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected.yaml` |
| 9 | SFT on rejected+chosen (resume) | BC6i Q4 | `llama-3-8b-base-ours4-6-sorted-score-diff-full-rebuttal-difficult-sft-rejected-and-chosen.yaml` |

## Key Arguments (no new experiments needed)

- **Tgvs Q1 (DPO+SFT baseline)**: Already in paper Table 9 as "DPO+NLL" (4.25% vs MixDPO 14.42%)
- **Z34t W3 (beta not tuned)**: beta=0.01 is Alignment Handbook default; all baselines use same beta; MixDPO also not tuned
- **Z34t W4 (computation-free)**: Revise claim to "convenient when ratings available"
