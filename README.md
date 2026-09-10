# LLM Learning

> 记录并系统整理大语言模型（LLM）学习笔记：从数学与神经网络基础，到 Transformer、模型架构、训练、推理、对齐与工程实践。

## 🎯 项目目标

这个仓库不是零散知识点的堆积，而是一份可持续维护的 **LLM 学习知识库**：

- 用自己的语言解释核心概念；
- 公式、直觉、代码/结构图并重；
- 记录“容易混淆的点”和“为什么这样设计”；
- 从基础组件逐步连接到完整 LLM；
- 便于后续快速搜索、复习和补充。

## 🗺️ 学习地图

```text
数学 / 神经网络基础
        ↓
Transformer 基础组件
        ↓
LLM 架构设计
        ↓
预训练 / 数据 / 优化
        ↓
推理 / KV Cache / 并行
        ↓
对齐 / 评测
        ↓
工程实践 / 论文阅读
```

## 📚 目录

| 模块 | 内容 | 状态 |
|---|---|---|
| [00 · 学习路线](docs/00-roadmap/README.md) | 学习顺序、知识地图、复习方法 | 🟢 已建立 |
| [01 · 基础知识](docs/01-foundations/) | 激活函数、归一化、Embedding、损失函数等 | 🟢 进行中 |
| [02 · Transformer](docs/02-transformer/) | Attention、MHA/GQA/MQA、RoPE、FFN、Residual | 🟡 待补充 |
| [03 · LLM 架构](docs/03-llm-architecture/) | LLaMA、DeepSeek、MoE、Dense vs MoE 等 | 🟡 待补充 |
| [04 · 训练](docs/04-training/) | 数据、预训练、优化器、并行训练、Scaling Law | 🟡 待补充 |
| [05 · 推理](docs/05-inference/) | KV Cache、Prefill/Decode、量化、推理加速 | 🟡 待补充 |
| [06 · 对齐与评测](docs/06-alignment-evaluation/) | SFT、RLHF、DPO、Reward Model、Benchmark | 🟡 待补充 |
| [07 · 工程实践](docs/07-engineering/) | vLLM、Transformers、部署、性能分析 | 🟡 待补充 |
| [08 · 论文笔记](docs/08-papers/) | 经典与最新论文的结构化阅读笔记 | 🟡 待补充 |

## ⭐ 当前重点笔记

### [激活函数与门控 FFN：ReLU、GELU、SiLU、SwiGLU](docs/01-foundations/activation-functions.md)

包含：

- Sigmoid / Tanh / ReLU / GELU / SiLU 对比；
- 标准正态分布中的 `φ(x)` 与 `Φ(x)`；
- 为什么 GELU 是 `xΦ(x)`；
- SiLU / Swish 的直觉；
- GLU、GEGLU、SwiGLU 的门控结构；
- 为什么 LLaMA 类模型常用 SwiGLU；
- 为什么 SwiGLU 的 FFN hidden size 常接近 `8/3 × d_model`，而不是传统的 `4 × d_model`。

## 🧠 推荐的笔记写法

每个知识点尽量保持同一结构：

1. **一句话定义**：先说明它是什么；
2. **公式**：给出最核心数学表达；
3. **直觉**：说明公式在“做什么”；
4. **结构图**：尽可能展示数据流；
5. **与相近概念对比**：例如 GELU vs SiLU；
6. **LLM 中的实际用途**：在哪些模块出现；
7. **容易混淆的点**：主动记录坑；
8. **进一步问题**：留下下一步学习入口。

## 🏷️ 状态标记

- 🟢 已学习 / 已整理
- 🟡 待补充
- 🔵 正在学习
- 🔴 需要重新理解

---

持续更新中。目标不是“看过”，而是逐步做到：**能解释、能推导、能连接到真实模型实现。**
