# DeepSeek-V3.2

> DeepSeek-V3.2 最值得从 Transformer 角度学习的是 **DeepSeek Sparse Attention（DSA）**：在 V3 系 MoE/MLA 基座上进一步降低长上下文 Attention 成本。

## 1. 一句话定位

V3.2 继续沿用 DeepSeek-V3 系列的大规模 MoE 主干，并引入：

\[
\boxed{\text{DeepSeek Sparse Attention (DSA)}}
\]

目标是在长上下文场景中减少 Attention 计算，同时尽量保留模型能力。

## 2. 为什么还需要 Sparse Attention

即使 MLA 已经显著压缩 KV Cache，标准全量 Attention 的计算仍随序列长度快速增长。

如果序列长度为 \(n\)，全 Attention 的核心 pairwise 交互规模近似：

\[
O(n^2)
\]

长到几十万、上百万 token 后，Attention 计算本身会成为新的瓶颈。

因此思路变成：

```text
MLA：解决“历史 K/V 存太多”
DSA：解决“每个 Query 看太多历史位置”
```

两者解决的是相关但不同的问题。

## 3. DSA 的直觉

Sparse Attention 不再让每个 Query 与所有历史 token 完整交互，而是通过可学习的索引/选择机制找到更重要的一小部分位置。

```text
Query
  │
Indexer / Selector
  │
选出少量相关历史 token
  │
Sparse Attention
```

因此长上下文 Attention 从“全部看一遍”变成“先找重点，再精算”。

## 4. 和 MLA 的关系

不要把 DSA 理解成 MLA 的替代品。

可以按两个维度理解：

| 技术 | 主要优化对象 |
|---|---|
| MLA | K/V 的表示与 KV Cache |
| DSA | Query 需要实际参与计算的历史 token 数量 |

所以 DeepSeek 的 Attention 优化路线可以粗略记成：

```text
GQA
 ↓
MLA：压缩 KV
 ↓
DSA：进一步稀疏长上下文 Attention
```

## 5. V3.2 的其他重点

V3.2 同时强化了：

- scalable RL；
- Agent 任务训练；
- reasoning + tool use。

但如果从 `02-transformer` 的学习目标出发，最优先掌握的仍然是 DSA。

## 6. 参考

- [DeepSeek-V3.2 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.2)
