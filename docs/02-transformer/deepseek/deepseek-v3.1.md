# DeepSeek-V3.1

> DeepSeek-V3.1 不是一次彻底的 Transformer 架构重写，而是在 V3 系列基座上继续做长上下文扩展与后训练升级。

## 1. 一句话定位

V3.1 的代表性变化是：

- 一个模型支持 Thinking / Non-Thinking 两种模式；
- Tool Calling / Agent 能力强化；
- 继续扩展长上下文训练；
- 底层仍属于 V3 系 MLA + MoE 主线。

## 2. Hybrid Thinking

V3.1 支持通过 chat template 在同一模型中切换：

```text
Thinking Mode
Non-Thinking Mode
```

这意味着“是否显式进行长推理”开始更多由后训练与交互格式控制，而不一定需要维护两个完全独立的模型。

## 3. 长上下文扩展

官方说明中，V3.1-Base 建立在原始 V3 checkpoint 之上，通过两阶段 long-context extension 继续训练。

因此它很适合作为一个案例理解：

> **上下文长度提升不一定意味着 Transformer 主干结构改变，也可能主要来自继续训练与位置编码扩展。**

## 4. 与 V3 的关系

```text
V3
MLA + DeepSeekMoE + MTP
 │
 └─ continued pretraining / long-context extension
      + post-training
      ↓
V3.1
Hybrid Thinking + stronger tool use
```

## 5. 学习重点

V3.1 建议重点关注：

- 架构升级 vs 后训练升级的区别；
- Chat Template 对 Thinking Mode 的控制；
- 长上下文 continued pretraining；
- Tool / Agent post-training。

## 6. 参考

- [DeepSeek-V3.1 model card](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)
- [DeepSeek V3.1 官方发布说明](https://api-docs.deepseek.com/zh-cn/news/news250821/)
