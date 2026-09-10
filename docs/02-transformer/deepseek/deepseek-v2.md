# DeepSeek-V2

> DeepSeek-V2 是 DeepSeek 架构路线真正发生明显变化的一代：**Attention 引入 MLA，FFN 引入 DeepSeekMoE。**

## 1. 一句话定位

DeepSeek-V2 是一个 236B 总参数、每个 token 激活约 21B 参数的 MoE 模型，支持 128K 上下文。最值得学习的两个结构是：

\[
\boxed{\text{MLA} + \text{DeepSeekMoE}}
\]

## 2. MLA：Multi-head Latent Attention

普通 Attention 在自回归推理时需要缓存历史 token 的 K/V：

```text
h_t → K_t, V_t → KV Cache
```

MLA 的核心思想是先把 K/V 信息压缩到低维 latent：

\[
c_t^{KV}=W^{DKV}h_t
\]

推理时主要缓存压缩后的 latent，而不是完整 K/V 表示，从而降低 KV Cache。

直觉上：

```text
标准 MHA/GQA：
历史 token → 完整 K/V → cache

MLA：
历史 token → 低秩压缩 latent → cache
                         ↓
                    Attention 使用
```

官方论文报告，相比 DeepSeek 67B，V2 的 KV Cache 可大幅下降，同时显著提升生成吞吐。

## 3. MLA 与 GQA / MQA 的区别

| 方法 | 思路 |
|---|---|
| MHA | 每个 Q head 都配自己的 K/V |
| GQA | 多个 Q head 共享一组 K/V |
| MQA | 基本所有 Q heads 共享 K/V |
| MLA | **把 K/V 信息压缩到 latent space** |

所以 MLA 不只是“减少 KV heads”，而是重新设计 K/V 的表示与缓存方式。

## 4. RoPE 与 MLA

MLA 中 Q/K 会区分内容部分与 RoPE 位置部分，可粗略理解为：

```text
Q = [content part, RoPE part]
K = [content part, RoPE part]
```

这样可以兼顾：

- 低秩压缩；
- 位置编码；
- 推理阶段的矩阵吸收与 KV Cache 压缩。

后面学习 MLA 时需要重点理解：**为什么位置相关维度不能简单全部塞进压缩 latent。**

## 5. DeepSeekMoE

普通 Dense FFN 对每个 token 都计算整套 FFN 参数。

MoE 则有多个专家：

```text
                 ┌→ Expert 1
Token → Router ──┼→ Expert 2
                 ├→ Expert 3
                 └→ ...
```

每个 token 只选择少数专家，因此可以同时做到：

- 总参数规模很大；
- 单 token 实际计算量远小于总参数量。

DeepSeekMoE 进一步区分 shared experts 与 routed experts，并通过更细粒度专家划分提高专家专业化程度。

## 6. V2 的意义

DeepSeek 后面的 V3、R1 等都建立在 V2 奠定的核心架构思路之上。

可以把 V2 看作：

\[
\boxed{\text{DeepSeek 从“标准 Transformer”进入“自有架构路线”的起点}}
\]

## 7. 参考

- [DeepSeek-V2 paper](https://arxiv.org/abs/2405.04434)
