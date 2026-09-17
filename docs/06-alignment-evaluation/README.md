# 06 · 对齐与评测

本目录记录大语言模型后训练与能力评估相关知识。

## 重点主题

- Supervised Fine-Tuning（SFT）
- Reinforcement Learning（Policy Gradient / PPO / GRPO）
- Preference Optimization
- Reward Modeling
- Agentic RL / Credit Assignment
- 模型评测方法
- Benchmark 设计
- LLM-as-a-Judge

## 已整理

### [从 Cross Entropy 到 LLM 强化学习](from-cross-entropy-to-rl.md)

从 next-token prediction 出发，逐步解释：

- `log_prob` 与 Cross Entropy 的关系；
- 没有 ground-truth next token 时 RL 如何构造 loss；
- Reward / Advantage 如何控制 token 概率上升或下降；
- 为什么 reward 不需要可微；
- 为什么 RL policy loss 可以为负；
- Policy Gradient 的推导；
- PPO ratio / clipping；
- GRPO group-relative advantage；
- 长 Agent trajectory 的 credit assignment 问题。

### [MiMo-V2.6 RL：大规模 Agentic RL 训练机制](mimo-v2.6-rl.md)

结合 MiMo-V2.6 实时 RL dashboard 与 MiMo-V2-Flash 技术报告，整理：

- `1,568 prompts × 16 rollouts` 的含义；
- reward distribution；
- Dynamic Sampler / Data Scheduler；
- Fully Async RL；
- partial rollout 与 policy staleness；
- test-case / rubric-based reward；
- Agentic In-group Credit Assignment；
- multi-task agentic RL；
- 约 2B tokens / step 的规模；
- 官方确认信息与尚未公开细节的边界。

## 学习目标

1. 理解基础模型如何经过后训练变成可用助手；
2. 从 Cross Entropy 连到 Policy Gradient，而不是把 RL 当成完全独立的一套反向传播；
3. 理解常见偏好优化与 RL 方法之间的区别；
4. 理解 Reward、Advantage、Credit Assignment 和 Sampling 对 RL 训练的影响；
5. 理解 Benchmark 指标能说明什么、不能说明什么；
6. 学会区分模型能力、对齐程度与实际使用体验。

## 待整理

- [ ] SFT 完整训练流程
- [ ] Reward Model
- [ ] PPO vs GRPO vs DAPO
- [ ] DPO / Preference Optimization
- [ ] Agent RL Credit Assignment
- [ ] Benchmark 方法论
- [ ] LLM-as-a-Judge
