# UltraInteract_pair 实验规划

## 数据集概览

- **来源**: `openbmb/UltraInteract_pair`，MIT license
- **论文**: Eurus (arXiv:2404.02078, ICLR 2025)
- **总量**: 219,522 pairs
- **格式**: 纯 binary labels (chosen/rejected)，无数值评分
- **字段**: `task`, `dataset`, `trajectory`, `chosen`, `rejected`, `id`, `parent_id`

## 数据分布

### 按 Task × Dataset 交叉统计

| Task | Dataset | 样本数 |
|------|---------|--------|
| **Math_CoT** | MATH | 25,765 |
| | mathqa | 17,743 |
| | gsm8k | 10,862 |
| | numglue | 2,861 |
| | **小计** | **57,231** |
| **Math_PoT** | MATH | 22,905 |
| | mathqa | 15,079 |
| | gsm8k | 10,257 |
| | tabmwp | 4,135 |
| | numglue | 3,467 |
| | **小计** | **55,843** |
| **Coding** | TACO | 50,877 |
| | codecontest | 44,319 |
| | wiki_table_questions | 1,544 |
| | **小计** | **96,740** |
| **Logic** | reclor | 7,958 |
| | hotpotqa | 1,009 |
| | strategyqa | 741 |
| | **小计** | **9,708** |

### 按 Task 汇总

| Task | 样本数 | 占比 |
|------|--------|------|
| Coding | 96,740 | 44.1% |
| Math_CoT | 57,231 | 26.1% |
| Math_PoT | 55,843 | 25.4% |
| Logic | 9,708 | 4.4% |

## Reviewer 问题

> How would the method work in the binary-label regime (no numeric rating scores), for example on verifiable math tasks, e.g., subsets of openbmb/UltraInteract_pair? Would MixDPO still be applicable and beneficial?

核心：MixDPO 在没有 rating scores 的 binary-label 数据上是否有效。

## 实验设计

### 训练数据选择

需要避免训练-评估数据污染。两种方案：

| 方案 | 训练数据 | 训练量 | 评估 benchmark | 污染风险 |
|------|----------|--------|---------------|---------|
| A | MATH 来源 (Math_CoT + Math_PoT) | ~49K | GSM8K | 无 |
| B | 排除 MATH 和 gsm8k (mathqa + numglue + tabmwp) | ~43K | MATH + GSM8K | 无 |

**建议方案 A**：用 MATH 来源训练（~49K），GSM8K 评估。规模和主实验（61K UltraFeedback）接近。

### Difficulty Signal

数据集无 rating scores，需要用 reward model 生成：
1. 用 reward model（如 `RLHFlow/ArmoRM-Llama3-8B-v0.1`）给 chosen/rejected 打分
2. 计算 reward margin = chosen_score - rejected_score
3. 按 margin 排序，bottom 10% 标记为 `is_difficult`

### 训练实验

| 实验 | 方法 | 说明 |
|------|------|------|
| Baseline | DPO | 标准 DPO，全部数据 |
| Ours | MixDPO | margin 排序，difficult pairs 用 SFT |
| (可选) | SFT on chosen | 纯 SFT 对照 |

- Base model: `princeton-nlp/Llama-3-Base-8B-SFT`（和主实验一致）
- 超参数: lr=5e-7, β=0.01, 1 epoch（和主实验一致）
- loss_type: `noisy-tolerant-4-6-flag`

### 评估

- Benchmark: **GSM8K** (8-shot) + 可选 **MATH** (4-shot)
- 指标: accuracy (pass@1)
- 工具: `lm_eval`

## 注意事项

### DPO 在 math/verifiable tasks 上的已知问题

多篇论文发现 DPO 在推理任务上效果差甚至有害：
- **Eurus**: MATH 上 DPO 28.3% vs KTO 33.2% vs NCA 34.2%
- **Step-DPO**: Qwen2-7B MATH 上 DPO 仅 +0.2%
- **3D-Properties**: Off-policy DPO 比 base model 还差
- **Iterative RPO**: MATH 上 DPO 12.4% < few-shot CoT 12.5%

根本原因：
1. 梯度纠缠：正确/错误解法共享 token，惩罚错误连带惩罚正确步骤
2. 整序列拒绝：错误通常在中间步骤，前面的正确推理被误伤
3. Chosen likelihood 下降：likelihood displacement 在推理任务上尤其严重

### 对 MixDPO 的预期

MixDPO 对 difficult pairs 用 SFT → 可能缓解上述问题：
- 如果 MixDPO > DPO → 强论据（在 DPO 已知困难的领域仍然有效）
- 如果两个都差 → 可解释（verifiable tasks 不是 DPO 的最佳适用场景，属于 domain 限制）

### 时间估算

| 步骤 | 预估时间 |
|------|---------|
| 数据准备 + reward model 打分 | 2-3h |
| 训练 (DPO + MixDPO) | 3-5h |
| 评估 (GSM8K) | 1-2h |
| **合计** | **6-10h** |

## 已有的相关证据（不需要新实验）

Table 3 中 SimPO pipeline 已经使用 PairRM reward margin（而非 rating scores）作为 difficulty signal，MixDPO 在该设置下取得了好结果。这已经部分回答了 reviewer 的问题。
