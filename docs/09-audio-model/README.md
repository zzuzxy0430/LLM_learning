# 09 · Audio Model

> 记录语音生成模型中的离散语音表示、声学模型、生成式建模、流式推理与声码器等内容。

## 已整理笔记

### [StepAudio 3 Gen：离散自回归统一音频生成](stepaudio-3-gen.md)

围绕 StepAudio 3 Gen 的核心设计，梳理以下内容：

- 12.5 Hz、16×2048 RVQ 的 StepAudio Tokenizer；
- LLM 沿时间轴只预测 `c0`、轻量 Predictor 沿码本深度补 `c1...c15`；
- RVQ Predictor 使用当前音频位置 `h_t` 而不是显式读取全部历史 hidden states；
- text token 与 `c0` 共用主 LM Head 的联合输出空间；
- RVQ Adaptor 如何把完整 16 层声学信息重新注入 LLM；
- 四阶段 progressive pretraining、gradient detach 与 joint cool-down；
- 与 VALL-E / MusicGen 以及 Diffusion / Flow Matching 路线的关系；
- 数据规模、GRPO、实验结果与当前开源状态。

### [VoxCPM：Tokenizer-Free TTS 的分层语义-声学建模](voxcpm.md)

围绕 VoxCPM 的 tokenizer-free 与 hierarchical semantic-acoustic modeling，重点整理：

- AudioVAE continuous latent、TSLM、FSQ、RALM 与 LocDiT 的完整链路；
- 为什么 `tokenizer-free` 并不等于完全 pure continuous；
- FSQ 作为 semi-discrete bottleneck 的真实作用；
- 历史连续 acoustic latent 如何经 LocEnc 得到 `E_<i`，重新反馈给后续生成；
- 去掉 FSQ、RALM、`E_<i`、`h_residual` 的关键消融结果；
- 为什么论文不能把全部收益简单归因于“没有 tokenizer”；
- VoxCPM 与 StepAudio 3 Gen 在 coarse semantic path + fine acoustic residual path 上的共性与差异。

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
