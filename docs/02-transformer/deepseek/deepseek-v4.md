# DeepSeek-V4

> DeepSeek-V4 是 DeepSeek 从 V2/V3 的 MLA 路线继续向 **百万 token 长上下文高效 Transformer** 演进的一代。

## 1. 一句话定位

V4 仍然是 Decoder-only MoE Transformer，但 Attention 和残差连接都发生了较明显变化。当前技术报告的核心关键词可以记成：

\[
\boxed{\text{Hybrid Attention} + \text{Compressed Attention} + \text{mHC} + \text{MoE}}
\]

官方公开的 V4 系列包含不同规模，其中 V4-Flash 约 284B 总参数、13B 激活参数；V4-Pro 规模更大。两者都面向 1M context。

## 2. 从 V3 到 V4：Attention 主线发生变化

V3 的核心是 MLA：

```text
V3
Multi-head Latent Attention
→ 压缩 KV 表示
```

V4 技术报告则转向混合 Attention 设计，把局部计算与高压缩长程信息结合起来。

官方技术描述中包括：

- **Compressed Sparse Attention (CSA)**
- **Heavily Compressed Attention (HCA)**
- Sliding-window / local attention

可以先形成这样的直觉：

```text
短距离依赖
   ↓
Local / Sliding Attention

长距离依赖
   ↓
Compressed / Sparse long-range path
```

目标是避免在 1M token 上对所有位置都进行昂贵的全量 Attention。

## 3. 为什么 V4 不只是“更大 KV Cache 优化”

长上下文有两个主要瓶颈：

1. **缓存问题**：历史 token 的 K/V 占显存；
2. **计算问题**：Query 和大量历史 token 做 Attention 很贵。

V2/V3 的 MLA 重点解决前者，同时改善计算效率；V3.2/V4 开始更明显地向稀疏、压缩 Attention 演进，以进一步处理百万上下文下的计算成本。

可以把路线粗略记成：

```text
GQA
 ↓
MLA：压缩 K/V
 ↓
DSA：稀疏选择相关位置
 ↓
V4 Hybrid Attention：局部 + 压缩长程信息
```

## 4. mHC：Manifold-Constrained Hyper-Connections

V4 还引入了 **mHC（Manifold-Constrained Hyper-Connections）**，用于替代/增强传统残差连接的信息流。

传统 Transformer 残差大致是：

\[
y=x+F(x)
\]

Hyper-Connection 的思想是让跨层信息流更加丰富，而 mHC 再通过约束提高深层网络训练时的稳定性。

这里暂时先记概念：

> V4 不只改 Attention，也开始重新设计 Transformer Block 的跨层连接方式。

后续可单独写 `mHC.md` 做数学展开。

## 5. MoE 仍然存在

V4 仍是 MoE 路线。以 V4-Flash 公开配置为例，可看到：

- 256 routed experts；
- 1 shared expert；
- 每 token 激活 6 个 routed experts；
- SiLU / SwiGLU 风格专家 FFN。

因此 DeepSeek 的大模型容量扩展主线仍然是：

\[
\boxed{\text{Sparse MoE}}
\]

## 6. V4 最值得学习的三个问题

1. CSA / HCA 和 V3 的 MLA 到底有什么本质区别？
2. 1M context 下，Attention FLOPs 和 KV Cache 分别如何下降？
3. mHC 为什么能比简单 residual connection 提供更强的信息流而又保持稳定？

## 7. 参考

- [DeepSeek-V4-Flash official model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Hugging Face DeepSeek-V4 architecture docs](https://huggingface.co/docs/transformers/model_doc/deepseek_v4)
- [DeepSeek V4 official API announcement](https://api-docs.deepseek.com/zh-cn/news/news260424/)
