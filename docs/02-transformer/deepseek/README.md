# DeepSeek Transformer 系列

> 这一目录按版本记录 DeepSeek 模型的 Transformer 架构演进。重点不是只记 benchmark，而是回答：**这一代相对上一代改了 Transformer 的什么？为什么？**

## 版本索引

| 版本 | 架构主线 | 最值得学的变化 | 笔记 |
|---|---|---|---|
| DeepSeek LLM / V1 | Dense Transformer | LLaMA 风格、GQA、RMSNorm、SiLU | [V1](deepseek-v1.md) |
| DeepSeek-V2 | MoE Transformer | **MLA + DeepSeekMoE** | [V2](deepseek-v2.md) |
| DeepSeek-V2.5 | V2 架构 | 通用 + Code 能力合并，主要是后训练升级 | [V2.5](deepseek-v2.5.md) |
| DeepSeek-V3 | MoE Transformer | MLA + DeepSeekMoE + **无辅助损失负载均衡 + MTP** | [V3](deepseek-v3.md) |
| DeepSeek-R1 | V3 系架构 | 架构不是重点，核心是 **RL 驱动推理能力** | [R1](deepseek-r1.md) |
| DeepSeek-V3.1 | V3 系架构 | Thinking / Non-Thinking 合一、长上下文扩展 | [V3.1](deepseek-v3.1.md) |
| DeepSeek-V3.2 | V3 系 MoE | **DeepSeek Sparse Attention (DSA)**、Agent / RL | [V3.2](deepseek-v3.2.md) |
| DeepSeek-V4 | 新一代 MoE | 1M context、压缩/稀疏注意力、mHC 等新设计 | [V4](deepseek-v4.md) |
| DeepSeek-V4.1-Flash | V4.1 新架构族 | 原生多模态、Flash 定位；内部结构待更多官方材料确认 | [V4.1 Flash](deepseek-v4.1-flash.md) |

## 一条线理解架构演进

```text
V1
Dense Transformer / GQA
      │
      ▼
V2
MLA + DeepSeekMoE
      │
      ▼
V3
MLA + DeepSeekMoE
+ Auxiliary-loss-free load balancing
+ MTP
      │
      ├────────► R1：重点转向 RL 推理训练
      │
      ▼
V3.1
Hybrid Thinking + 更长上下文后训练
      │
      ▼
V3.2
DeepSeek Sparse Attention
      │
      ▼
V4
面向 1M context 的新一代高效 Transformer
      │
      ▼
V4.1 Flash
新架构族 + 原生多模态
```

## 学习时重点观察四个位置

每一代 DeepSeek 都建议从下面四个位置对比：

1. **Attention**：MHA/GQA → MLA → Sparse / Compressed Attention；
2. **FFN**：Dense SwiGLU → DeepSeekMoE；
3. **Residual / Norm**：RMSNorm、Residual 到后续连接结构；
4. **Training Objective / Post-training**：MTP、RL、Thinking/Non-Thinking、Agent 训练。

## 说明

- `V1` 是这里为了学习方便对早期 DeepSeek LLM 系列的称呼；官方模型名主要是 DeepSeek LLM 7B / 67B。
- `R1` 严格来说不是 V3.x 的“架构版本”，但它对 DeepSeek 的推理训练路线非常关键，所以单独保留一篇。
- V4.1-Flash 于 2026-09-10 发布；当前笔记只记录已经能可靠确认的信息，未公开或未验证的内部结构不会猜测。
