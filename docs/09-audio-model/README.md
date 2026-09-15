# 09 · Audio Model

> 记录语音生成模型中的离散语音表示、声学模型、生成式建模、流式推理与声码器等内容。

## 已整理笔记

### [CosyVoice 3 中 Flow Matching 与 DiT 的原理](cosyvoice3-flow-matching-dit.md)

从 CosyVoice 3 源码出发，梳理以下内容：

- LLM speech token、Flow Matching、DiT 与 HiFT vocoder 的完整链路；
- Conditional Flow Matching 的训练路径、速度目标和 ODE 推理；
- 时间条件通过 AdaLayerNorm 注入 DiT 的方式；
- `mu`、prompt mel 和 speaker embedding 的逐帧特征拼接；
- 25 Hz speech token 与 50 Hz mel 的长度对齐；
- prompt token 与 LLM 生成 token 的拼接；
- prompt mel 的 prefix acoustic conditioning；
- Classifier-Free Guidance 与流式 chunk attention。

也提供一份可下载后直接用浏览器打开的 [HTML 版本](cosyvoice3-flow-matching-dit.html)。

## 后续可补充

- Speech tokenizer / neural codec
- Diffusion、Flow Matching 与 Rectified Flow 对比
- Vocoder：HiFi-GAN、HiFT、BigVGAN
- 流式 TTS 的延迟与缓存设计
- CosyVoice、VALL-E、Voicebox、F5-TTS 等模型对比
