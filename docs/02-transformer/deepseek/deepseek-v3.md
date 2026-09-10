# DeepSeek-V3

> DeepSeek-V3 延续 V2 的 MLA + DeepSeekMoE 主线，但把训练效率、MoE 负载均衡和训练目标进一步推进。

## 1. 一句话定位

DeepSeek-V3 是一个 671B 总参数、每 token 激活约 37B 参数的 MoE 模型。架构主线可以记成：

\[
\boxed{\text{MLA} + \text{DeepSeekMoE} + \text{Auxiliary-loss-free load balancing} + \text{MTP}}
\]

## 2. MLA 继续保留

V3 继续使用 Multi-head Latent Attention，目标仍然是降低 KV Cache 和提升推理效率。

因此 DeepSeek 的 Attention 演进到 V3 时并不是重新发明一套机制，而是继续沿用 V2 已验证的 MLA。

## 3. DeepSeekMoE 继续扩展

V3 继续采用稀疏 MoE：

```text
Token
  │
Router
  │
Top-k Experts
  │
Weighted Sum
```

大总参数量与较低激活参数量并存，使模型能够扩大容量而不让每个 token 都承担全部计算。

## 4. 无辅助损失负载均衡

传统 MoE 往往会增加 auxiliary loss 来避免所有 token 都挤到少数专家。

V3 的一个重要改进是使用 **auxiliary-loss-free load balancing** 策略。

核心目标是：

> 在不额外加入会干扰主训练目标的辅助损失情况下，让专家负载仍然保持相对均衡。

这一点值得和传统 MoE Router 的 load-balancing loss 单独对比学习。

## 5. MTP：Multi-Token Prediction

传统语言模型预训练通常预测下一个 token：

\[
P(x_{t+1}\mid x_{\le t})
\]

V3 引入 Multi-Token Prediction 训练目标，让模型在训练阶段额外预测未来多个 token。

直觉上：

```text
传统：当前位置 → 预测 t+1
MTP：当前位置 → 同时学习 t+1、t+2、... 的未来信息
```

它的目的不是简单让推理一次吐多个 token，而是通过额外训练信号改善表示学习和预测能力。

## 6. 为什么 V3 很重要

V3 把 DeepSeek-V2 奠定的架构做成了一个更成熟、更大规模的系统：

```text
V2：提出 MLA + DeepSeekMoE
V3：在大规模训练中进一步优化 MoE、训练目标和系统效率
```

而 DeepSeek-R1 的基础模型路线也与 V3 密切相关，因此理解 V3 是理解 R1 的前提。

## 7. 参考

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
