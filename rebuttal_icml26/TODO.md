# Rebuttal TODO List

## Training Experiments ✅ ALL DONE

- [x] **beta-dpo** — Z34t W2: β-DPO baseline
- [x] **dpo-beta005** — Z34t W3: DPO β=0.05
- [x] **dpo-beta01** — Z34t W3: DPO β=0.1
- [x] **mixdpo-beta005** — Z34t W3: MixDPO β=0.05
- [x] **mixdpo-beta01** — Z34t W3: MixDPO β=0.1
- [x] **mixdpo-sorted-chosen-score** — Tgvs Q2: sorted by chosen score
- [x] **ours4-6-...-rerun** — Z34t Q3: rerun for checkpoint-336
- [x] **rebuttal-sft-rejected** — BC6i Q4: SFT on rejected only
- [x] **rebuttal-sft-rejected-and-chosen** — BC6i Q4: SFT on both

## AlpacaEval Evaluation (8 models)

- [x] beta-dpo ✅
- [x] dpo-beta005 ✅
- [x] dpo-beta01 ✅
- [x] mixdpo-beta005 ✅
- [ ] mixdpo-beta01 ⚠️ incomplete (354/805), needs rerun
- [ ] mixdpo-sorted-chosen-score (needs rerun with noisy-tolerant-4-6-flag)
- [ ] mixdpo-sorted-rejected-score (new)
- [x] sft-rejected (new version) ✅
- [x] sft-rejected-and-chosen ✅

## Analysis ✅ ALL DONE

- [x] **Arena-Hard 95% CI** (Z34t Q1) — LLaMA-3-8B done
- [x] **Table 14 stderr** (Z34t W1) — lm_eval built-in stderr
- [x] **Likelihood displacement** (Z34t Q3) — eval logps table
- [x] **Pre-training NLL distribution** (Z34t Q4) — per-token NLL, 500 samples/subset
- [x] **Length statistics** (Z34t Q5) — DPO stage vs SFT stage token lengths
- [x] **Qwen2.5-7B AlpacaEval SE** (Z34t Q6) — AlpacaEval built-in SE
- [x] **Downstream task results with stderr** — collected via lm_eval

## Fill Results into Rebuttal (after AlpacaEval)

- [x] Z34t W2: β-DPO baseline results ✅
- [x] Z34t W3: β sweep table ✅ (mixdpo-beta01 pending rerun)
- [ ] Tgvs Q2: mixdpo-sorted-chosen-score + sorted-rejected-score results
- [x] BC6i Q4: SFT on chosen/rejected/both comparison (fair, from same checkpoint) ✅

## Still TODO

- [x] **Qwen2.5-7B Arena-Hard** (Z34t Q6) ✅
- [ ] **Update Q3 with compute_eval_logps.py results** — add Mistral/Qwen data to displacement table ⚠️ current results have issues, needs rerun
- [x] **Binary-label experiment** (Z34t Q7) — addressed with SimPO/PairRM reward margin results (Table 3)

## Paper Text Updates ✅ ALL DONE

- [x] Add 6 related works discussion (Z34t W5)
- [x] Revise "computation-free" claim (Z34t W4)
- [x] Clarify DPO+NLL is DPO+SFT baseline (Tgvs Q1)
- [x] Clarify contributions (BC6i Q2/Q3)
- [x] Add soft-weighting discussion (LcVx Q1)
- [x] Add binary-preference discussion (LcVx Q2)
