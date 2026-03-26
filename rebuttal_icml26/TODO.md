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

- [ ] beta-dpo
- [ ] dpo-beta005
- [ ] dpo-beta01
- [ ] mixdpo-beta005
- [ ] mixdpo-beta01
- [ ] mixdpo-sorted-chosen-score
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

- [ ] Z34t W2: β-DPO baseline results
- [ ] Z34t W3: β sweep table (DPO vs MixDPO at β=0.01/0.05/0.1)
- [ ] Tgvs Q2: mixdpo-sorted-chosen-score results
- [x] BC6i Q4: SFT on chosen/rejected/both comparison (fair, from same checkpoint) ✅

## Still TODO

- [x] **Qwen2.5-7B Arena-Hard** (Z34t Q6) ✅
- [ ] **Likelihood displacement with denser eval** — redraw with dpo-beta01/mixdpo-beta01 wandb data (~9 eval points)
- [ ] **Update Q3 likelihood displacement table** — use rerun (eval_steps=40) wandb data to replace current 4-point data
- [x] **Binary-label experiment** (Z34t Q7) — addressed with SimPO/PairRM reward margin results (Table 3)

## Paper Text Updates ✅ ALL DONE

- [x] Add 6 related works discussion (Z34t W5)
- [x] Revise "computation-free" claim (Z34t W4)
- [x] Clarify DPO+NLL is DPO+SFT baseline (Tgvs Q1)
- [x] Clarify contributions (BC6i Q2/Q3)
- [x] Add soft-weighting discussion (LcVx Q1)
- [x] Add binary-preference discussion (LcVx Q2)
