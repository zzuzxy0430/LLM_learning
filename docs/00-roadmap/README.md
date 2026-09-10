# 00 · LLM 学习路线

这部分用于维护整个仓库的学习顺序，避免知识点越记越散。

## 第一阶段：神经网络与数学基础

目标：看懂 Transformer 中每一个基本算子。

建议顺序：

1. 线性层 / MLP
2. 激活函数
3. Softmax
4. Cross Entropy
5. Embedding
6. LayerNorm / RMSNorm
7. Residual Connection
8. 初始化与梯度传播

当前笔记：

- [激活函数与门控 FFN](../01-foundations/activation-functions.md)

## 第二阶段：Transformer

目标：能从输入 token 一路解释到 Transformer Block 输出。

建议顺序：

1. Self-Attention
2. Q / K / V 的含义
3. Scaled Dot-Product Attention
4. Multi-Head Attention
5. MQA / GQA
6. Positional Encoding / RoPE
7. FFN
8. Residual + Norm
9. Pre-Norm vs Post-Norm
10. 完整 Transformer Block

## 第三阶段：现代 LLM 架构

目标：理解 LLaMA、DeepSeek 等模型为什么这样设计。

重点：

- RMSNorm
- SwiGLU
- RoPE
- GQA
- Dense vs MoE
- Expert Routing
- MLA
- Long Context

## 第四阶段：训练

目标：理解“模型是怎么被训练出来的”。

重点：

- Tokenizer
- 数据清洗 / 去重 / 混合
- Pretraining Objective
- AdamW
- Learning Rate Schedule
- Gradient Accumulation
- Mixed Precision
- Data / Tensor / Pipeline Parallelism
- ZeRO / FSDP
- Scaling Law

## 第五阶段：推理

目标：理解为什么 LLM 推理贵，以及各种推理优化在优化什么。

重点：

- Autoregressive Decoding
- Prefill vs Decode
- KV Cache
- Continuous Batching
- PagedAttention
- FlashAttention
- Quantization
- Speculative Decoding
- Tensor Parallel Inference

## 第六阶段：对齐与评测

重点：

- SFT
- Reward Model
- RLHF
- PPO
- DPO
- GRPO
- Process Reward / Outcome Reward
- Benchmark
- LLM-as-a-Judge

## 第七阶段：工程与论文

最终目标：

- 能读 Hugging Face 模型配置与源码；
- 能从论文快速定位模型创新点；
- 能根据 profiler 判断训练 / 推理瓶颈；
- 能把数学公式与 PyTorch 实现对应起来；
- 能解释主流模型架构之间的差异。

## 学习原则

每学一个概念，至少回答四个问题：

1. **它是什么？**
2. **为什么需要它？**
3. **它具体怎么算？**
4. **真实 LLM 里在哪里使用？**

如果还能回答“它有什么替代方案、为什么不用别的”，说明基本理解到位。
