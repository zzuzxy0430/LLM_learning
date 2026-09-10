# DeepSeek-R1

> DeepSeek-R1 对学习者最重要的地方不是“又换了一套 Transformer Block”，而是展示了 **如何用强化学习把已有强基座模型训练成推理模型**。

## 1. 一句话定位

R1 延续 DeepSeek 的 MoE / MLA 基座路线，但真正的创新重点在 **Reasoning Post-training**，尤其是强化学习。

所以建议把它放在 DeepSeek 架构目录里，但学习重点标成：

\[
\boxed{\text{Transformer Architecture} \; + \; \text{Reasoning RL}}
\]

## 2. 为什么 R1 不应该和 V2/V3 混为一谈

V2/V3 的核心问题是：

> Attention、FFN、MoE、KV Cache 怎么设计得更高效？

R1 更关注：

> 已经有一个很强的语言模型之后，怎样让它学会长链条推理、反思和验证？

因此：

```text
V2/V3 → 架构创新
R1    → 推理训练范式创新
```

## 3. R1-Zero 的意义

R1 工作展示了一个很重要的现象：在没有先进行大量人工标注 CoT SFT 的情况下，强化学习也能诱导模型出现：

- 长链条推理；
- 自我检查；
- 反思与修正；
- 更强的数学/代码问题求解能力。

随后 R1 再结合冷启动数据和多阶段训练，提高可读性、稳定性与综合能力。

## 4. 和 Transformer 的连接

从 Transformer 结构本身看，R1 并不是理解 MLA/MoE 的新起点。

学习时更适合这样连接：

```text
DeepSeek-V3
   │
   ├─ 架构：MLA + DeepSeekMoE
   │
   └─ 后训练
        ↓
   DeepSeek-R1
        ↓
   强化学习驱动推理能力
```

## 5. 后续需要单独学习

- GRPO / RL 训练思路；
- Reward 设计；
- R1-Zero vs R1；
- Reasoning token / long CoT；
- Distillation。

这些内容更适合在 `06-alignment-evaluation` 中进一步展开。

## 6. 参考

- [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948)
