# 02 · Transformer

本目录记录 Transformer 的核心组件、数据流，以及现代 LLM 对标准 Transformer 的架构改造。

## 建议学习顺序

1. Self-Attention
2. Q / K / V
3. Scaled Dot-Product Attention
4. Multi-Head Attention
5. MQA / GQA
6. Positional Encoding / RoPE
7. FFN / SwiGLU
8. Residual Connection
9. LayerNorm / RMSNorm
10. Pre-Norm vs Post-Norm
11. 完整 Transformer Block
12. 现代模型如何改造 Transformer

## DeepSeek Transformer 系列

已建立独立目录：**[deepseek/](deepseek/README.md)**。

按版本拆分，方便对照每一代到底改了 Transformer 的什么：

- [DeepSeek LLM / V1](deepseek/deepseek-v1.md)：Dense Transformer、GQA、RMSNorm、SiLU
- [DeepSeek-V2](deepseek/deepseek-v2.md)：**MLA + DeepSeekMoE**
- [DeepSeek-V2.5](deepseek/deepseek-v2.5.md)：沿用 V2 架构，通用/代码能力融合
- [DeepSeek-V3](deepseek/deepseek-v3.md)：MLA + DeepSeekMoE + 无辅助损失负载均衡 + MTP
- [DeepSeek-R1](deepseek/deepseek-r1.md)：V3 系架构上的强化学习推理路线
- [DeepSeek-V3.1](deepseek/deepseek-v3.1.md)：Hybrid Thinking、长上下文继续训练
- [DeepSeek-V3.2](deepseek/deepseek-v3.2.md)：**DeepSeek Sparse Attention (DSA)**
- [DeepSeek-V4](deepseek/deepseek-v4.md)：百万上下文、Hybrid/Compressed Attention、mHC、MoE
- [DeepSeek-V4.1-Flash](deepseek/deepseek-v4.1-flash.md)：2026-09-10 发布，新架构族、原生多模态；技术细节随官方材料继续补充

### DeepSeek 架构演进速记

```text
V1: Dense + GQA
      ↓
V2: MLA + DeepSeekMoE
      ↓
V3: MLA + MoE + MTP + 更成熟的负载均衡
      ↓
V3.2: DSA / Sparse Attention
      ↓
V4: 1M context + Hybrid/Compressed Attention + mHC
      ↓
V4.1: New architecture family + Native Multimodality
```

## 与基础知识的连接

- 激活函数与 FFN：见 [激活函数与门控 FFN](../01-foundations/activation-functions.md)
- DeepSeekMoE 里的专家 FFN 仍会用到 SiLU / SwiGLU 等概念。
- MLA / DSA / V4 Attention 都建立在 Q/K/V 与 Attention 基础之上，建议先学标准 Attention 再看版本演进。

## 待整理

- [ ] Attention 数学推导
- [ ] Q/K/V 的直觉解释
- [ ] MHA vs MQA vs GQA
- [ ] RoPE
- [ ] RMSNorm
- [ ] Transformer Block 完整数据流
- [ ] MLA 单独数学推导
- [ ] DeepSeekMoE Router / Shared Expert / Routed Expert
- [ ] DeepSeek Sparse Attention 详细推导
- [ ] V4 CSA / HCA / mHC 详细拆解
- [ ] V4.1-Flash 官方技术报告发布后补全 Block 结构
