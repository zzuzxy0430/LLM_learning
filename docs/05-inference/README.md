# 05 · 推理

本目录记录 LLM 推理阶段的计算流程、显存占用和性能优化。

## 重点主题

- Autoregressive Decoding
- Prefill vs Decode
- KV Cache
- KV Cache 内存计算
- Continuous Batching
- PagedAttention
- FlashAttention
- Quantization
- Speculative Decoding
- Tensor Parallel Inference
- Throughput vs Latency

## 建议重点回答

学习每种优化时，优先问：

1. 它优化的是计算、显存还是通信？
2. 主要作用于 Prefill 还是 Decode？
3. 会不会牺牲精度？
4. 对吞吐和单请求延迟分别有什么影响？

## 待整理

- [ ] KV Cache 原理与删除 / 滑窗机制
- [ ] Prefill vs Decode
- [ ] FlashAttention
- [ ] PagedAttention
- [ ] Speculative Decoding
- [ ] 量化方法对比
