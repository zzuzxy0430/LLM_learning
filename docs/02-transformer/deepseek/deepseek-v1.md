# DeepSeek LLM / V1

> 这里用 **V1** 指代 DeepSeek 最早期的通用 LLM 主线，即 DeepSeek LLM 7B / 67B。它还是比较标准的 Dense Decoder-only Transformer，是理解后续 V2/V3 架构变化的基线。

## 1. 一句话定位

DeepSeek LLM 67B 是一个 **LLaMA 风格的 Dense Transformer**，采用 GQA、RMSNorm、SiLU 等现代 Decoder-only LLM 常见组件。

## 2. 代表配置：DeepSeek LLM 67B

官方 Hugging Face 配置可见：

- hidden size：8192
- layers：95
- attention heads：64
- KV heads：8
- max position embeddings：4096
- activation：SiLU
- normalization：RMSNorm
- architecture：`LlamaForCausalLM`

这意味着它的 Attention 属于 **GQA**：

\[
64\text{ Query Heads} \quad / \quad 8\text{ KV Heads}
\]

即约 8 个 Query Heads 共享一组 K/V。

## 3. 为什么 V1 值得保留

后续 DeepSeek 的真正架构创新可以理解为相对这个基线逐步发生：

```text
V1: Dense FFN + GQA
        │
        ▼
V2: DeepSeekMoE + MLA
        │
        ▼
V3: MLA + 更成熟的 MoE + MTP
```

所以学习 V1 的目标不是研究某个独特模块，而是建立一个“标准现代 Transformer”参照物。

## 4. 与后续版本最关键区别

| 模块 | V1 | V2 以后 |
|---|---|---|
| Attention | GQA | MLA / Sparse Attention 等 |
| FFN | Dense | MoE |
| KV Cache 优化 | 主要靠 GQA | MLA 压缩 KV 表示 |
| 参数规模扩展 | Dense 扩展 | 稀疏专家扩展 |

## 5. 参考

- [DeepSeek LLM paper](https://arxiv.org/abs/2401.02954)
- [DeepSeek LLM 67B Base](https://huggingface.co/deepseek-ai/deepseek-llm-67b-base)
