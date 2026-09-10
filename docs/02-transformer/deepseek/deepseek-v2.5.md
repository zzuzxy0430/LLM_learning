# DeepSeek-V2.5

> DeepSeek-V2.5 的重点不是新的 Transformer 基础模块，而是把 DeepSeek-V2-Chat 与 DeepSeek-Coder-V2-Instruct 的能力融合到一个模型中。

## 1. 一句话定位

DeepSeek-V2.5 延续 V2 的核心架构：

\[
\boxed{\text{MLA} + \text{DeepSeekMoE}}
\]

主要升级发生在能力融合和后训练，而不是重新设计 Attention / FFN。

## 2. 和 V2 的关系

官方模型卡将 V2.5 描述为：

- 融合 DeepSeek-V2-Chat 的通用能力；
- 融合 DeepSeek-Coder-V2-Instruct 的代码能力；
- 改善人类偏好对齐、写作和指令遵循。

因此学习架构时可以把它记作：

```text
V2      → 架构创新：MLA + DeepSeekMoE
V2.5    → 沿用 V2 架构，强化后训练和能力融合
```

## 3. 代表配置

DeepSeek-V2.5 配置中仍可以看到 MLA / MoE 的典型字段：

```text
kv_lora_rank
q_lora_rank
qk_nope_head_dim
qk_rope_head_dim
n_routed_experts
n_shared_experts
num_experts_per_tok
```

这些字段很适合作为后续阅读模型源码时的入口。

## 4. 学习结论

V2.5 最重要的认知是：

> **模型“版本升级”不一定意味着 Transformer 架构升级。**

有时底层网络结构基本不变，但通过数据、SFT、偏好对齐、代码训练等手段，能力会明显变化。

这也是后续理解 V3.1、R1 等版本时非常重要的区分：

\[
\text{Architecture Change} \neq \text{Post-training Change}
\]

## 5. 参考

- [DeepSeek-V2.5 model card](https://huggingface.co/deepseek-ai/DeepSeek-V2.5)
- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434)
