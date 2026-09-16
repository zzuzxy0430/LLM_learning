# VoxCPM：Tokenizer-Free TTS 的分层语义-声学建模

> 重点理解 VoxCPM 的核心结构，以及一个容易被标题掩盖的问题：它的收益到底来自“tokenizer-free”，还是来自把连续声学历史重新反馈给语言模型并做残差声学建模？

## 1. 先给结论

VoxCPM 的核心不能简单概括成“去掉 speech tokenizer”。更准确的理解是：

> **高层用 semi-discrete 表示稳定 semantic / prosody planning，低层保留 continuous acoustic information，并通过 RALM 显式建模残差声学细节。**

因此它虽然叫 **Tokenizer-Free TTS**，但并不是“完全连续、完全没有量化”。它不依赖外部预训练 speech tokenizer，也不让主 LM 预测 codec token ID；但在 TSLM hidden state 上加入了 **FSQ（Finite Scalar Quantization）semi-discrete bottleneck**，作为内部的可微归纳偏置。

论文自己的消融反而说明：

- 完全去掉 FSQ、变成更纯的 continuous modeling，会明显恶化 hard-case 稳定性；
- 去掉 Residual Acoustic LM（RALM）也会明显恶化；
- **去掉 RALM 中的历史 acoustic embedding `E_<i`，退化尤其明显**。

所以更稳妥的归因是：

> **VoxCPM 的优势来自“semi-discrete semantic planning + continuous acoustic residual modeling + historical acoustic feedback”这一整套分层设计，而不能归因于 tokenizer-free 这一个变量。**

---

## 2. VoxCPM 的整体生成链路

传统 codec-LM TTS 常见的是：

```text
speech
  ↓
pretrained speech tokenizer
  ↓
discrete speech tokens
  ↓
LLM predicts token IDs
  ↓
codec decoder
  ↓
waveform
```

VoxCPM 则改成连续 latent 路线：

```text
Text + previous audio latents
            │
            ▼
      AudioVAE / LocEnc
            │
       acoustic embedding E
            │
            ▼
          TSLM
   semantic / prosody planning
            │
           FSQ
  semi-discrete bottleneck
            │
            ▼
          RALM  ◀──── historical acoustic E_<i
   residual acoustic modeling
            │
            ▼
          LocDiT
 local diffusion transformer
            │
            ▼
 continuous audio latent patch z_i
            │
            ▼
       AudioVAE Decoder
            │
            ▼
         waveform
```

可以把职责划分成：

- **TSLM**：负责长时语义、内容、节奏、prosody planning；
- **FSQ**：给 TSLM 的语音表示施加 semi-discrete bottleneck，避免 semantic 与 acoustic detail 在连续空间中完全纠缠；
- **RALM**：恢复 FSQ 压缩后不适合由高层承担的细粒度声学残差；
- **LocDiT**：根据上层条件，在连续 latent 空间生成局部音频 patch；
- **AudioVAE**：把 waveform 映射到/从连续 acoustic latent。

官方项目页将其概括为 hierarchical semantic-acoustic modeling：TSLM 负责 semantic-prosodic plan，RALM 恢复 fine-grained acoustic details，再共同指导 local diffusion decoder。

---

## 3. “Tokenizer-Free”到底是什么意思？

这里最容易误解。

VoxCPM 的 tokenizer-free 主要表示：

1. 不需要一个独立、预训练好的 speech tokenizer；
2. 不把 speech 压成一串离散 codec token ID，然后让 LLM 做 next-token classification；
3. 整个 TTS 模型可以围绕连续 AudioVAE latent 端到端训练。

但它并不是完全没有离散化约束。

### 3.1 FSQ 是一个 semi-discrete bottleneck

TSLM 的 speech hidden state 会经过 FSQ：

```text
TSLM hidden
    ↓
   FSQ
    ↓
semi-discrete representation
```

FSQ 的作用不是构造一个“speech vocabulary”让模型预测，而是把 hidden state 限制到有限格点附近，形成一种内部的 regularization / inductive bias。

所以更准确的对比是：

```text
传统 codec LM
Audio → tokenizer → discrete token ID → LM predicts token ID

VoxCPM
Audio → continuous latent → LM hidden → FSQ regularization
     → RALM / LocDiT → continuous latent
```

这也是为什么“tokenizer-free”不能直接等同于“pure continuous”。

---

## 4. 历史连续声学信息是怎么重新塞回模型的？

这是 VoxCPM 很关键的一条通路。

对于已经生成的历史 audio latent：

\[
Z_{<i} = (z_1, z_2, ..., z_{i-1})
\]

模型通过局部编码器得到历史 acoustic embedding：

\[
E_{<i} = \mathrm{LocEnc}(Z_{<i})
\]

这些 acoustic embeddings 并不会在生成下一步时被丢掉，而是重新作为上下文参与后续建模。

尤其在 RALM 中，输入不只是上层的 semi-discrete TSLM/FSQ representation，还显式包含历史 acoustic embedding `E_<i`。

可以画成：

```text
historical generated latent Z_<i
              │
              ▼
            LocEnc
              │
             E_<i
            /    \
           ▼      ▼
        TSLM     RALM
          │        │
         FSQ   acoustic residual
           \      /
            \    /
             LocDiT
               │
               ▼
              z_i
```

这意味着 VoxCPM 在自回归推进时，不只是依赖一个被压缩后的 semantic state；**之前已经生成出的真实连续声学状态，会被重新编码并反馈给后续生成。**

这种设计对以下信息尤其重要：

- speaker identity；
- timbre；
- accent；
- recording condition；
- micro-prosody；
- spectral details。

---

## 5. 为什么还需要 RALM？

如果只让一个 TSLM 同时承担：

```text
semantic planning
+ prosody
+ speaker identity
+ fine spectral detail
+ local acoustic continuity
```

连续空间里的任务很容易纠缠。

VoxCPM 的做法是显式拆成：

```text
TSLM + FSQ
    │
    └── coarse semantic / prosodic representation

RALM + E_<i
    │
    └── residual acoustic detail
```

最终再把两路表示结合起来作为 LocDiT 的条件。

官方的 t-SNE 分析也支持这种“自然分工”：TSLM-FSQ representation 更贴近 text / semantic-prosodic structure，而 RALM residual 对 speaker-related acoustic variation 更敏感。

---

## 6. 最关键的消融：收益到底来自哪里？

论文的 architecture ablation 非常有价值，因为它直接测试了 FSQ、RALM、历史 acoustic embedding 等部分。

### 6.1 默认配置

默认配置使用 `FSQ d256s9`：

| Setting | EN WER ↓ | EN SIM ↑ | ZH CER ↓ | ZH SIM ↑ | ZH-hard CER ↓ | ZH-hard SIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Default | 2.98 | 62.6 | 1.77 | 70.4 | 18.19 | 64.9 |

### 6.2 去掉 FSQ：pure continuous 并没有更好

| Setting | EN WER ↓ | ZH CER ↓ | ZH-hard CER ↓ |
|---|---:|---:|---:|
| Default (`FSQ d256s9`) | 2.98 | 1.77 | 18.19 |
| `w/o FSQ` | 3.67 | 2.30 | **24.92** |

特别是 hard case：

\[
18.19 \rightarrow 24.92
\]

说明完全去掉 semi-discrete bottleneck 后，长程/困难样本上的错误累积反而更严重。

这直接反驳了一个过度简化的说法：

> “VoxCPM 强，是因为 continuous 一定比 discrete 好。”

论文自身的结果更接近：**需要一定程度的离散化归纳偏置来稳定高层 planning，但又不希望外部 speech tokenizer 把所有信息都压成 codec token。**

### 6.3 去掉 RALM：不是单纯把 TSLM 做大就能补回来

| Setting | EN WER ↓ | ZH CER ↓ | ZH-hard CER ↓ |
|---|---:|---:|---:|
| Default | 2.98 | 1.77 | 18.19 |
| w/o RALM, 24-layer TSLM | 4.34 | 3.05 | 25.00 |
| w/o RALM, larger/scratch TSLM variant | 5.35 | 3.46 | 30.40 |

这说明 hierarchical division of labor 本身很重要，不只是“多加了一些参数”。

如果把 RALM 去掉，单流 TSLM 即使增加容量，也无法稳定替代 residual acoustic modeling。

### 6.4 去掉历史 acoustic embedding `E_<i`：退化非常明显

最能回答“是不是把原始/连续音频重新塞回模型带来的收益”的消融是：

`w/o E_<i in RALM`

| Setting | EN WER ↓ | EN SIM ↑ | ZH CER ↓ | ZH SIM ↑ | ZH-hard CER ↓ | ZH-hard SIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Default | 2.98 | 62.6 | 1.77 | 70.4 | 18.19 | 64.9 |
| w/o acoustic `E_<i` | **4.91** | **60.9** | **4.94** | **68.1** | **27.17** | **61.7** |

普通中文 CER：

\[
1.77 \rightarrow 4.94
\]

ZH-hard：

\[
18.19 \rightarrow 27.17
\]

speaker similarity 也下降。

这个结果很强地支持：

> **历史连续声学信息的显式反馈，是 VoxCPM 生成稳定性、speaker identity 和 fine acoustic detail 的重要来源。**

也就是说，VoxCPM 并不是把 acoustic information 全压缩进一个 high-level LM hidden state 后就不管了；它保留了一条 continuous acoustic shortcut / residual path。

### 6.5 去掉 RALM residual 对 LocDiT 的条件

论文还测试了 `w/o h_residual in condition`：

| Setting | EN WER ↓ | EN SIM ↑ | ZH CER ↓ | ZH-hard CER ↓ |
|---|---:|---:|---:|---:|
| Default | 2.98 | 62.6 | 1.77 | 18.19 |
| w/o `h_residual` | 3.86 | **58.3** | 3.05 | 23.65 |

尤其 similarity 下降明显，也说明 residual acoustic representation 对音色与声学保真度是实质性的。

---

## 7. 因此：能不能说收益主要来自 tokenizer-free？

**不能直接这么归因。**

论文没有做一个最干净的 controlled experiment：

```text
A. 完全相同的 VoxCPM architecture + continuous latent
B. 完全相同的 VoxCPM architecture + external discrete tokenizer
```

只改变 tokenizer 这一项，然后比较最终性能。

实际 VoxCPM 相比 codec-LM 同时改变了很多变量：

- external speech tokenizer → AudioVAE continuous latent；
- TSLM hidden 上增加 FSQ semi-discrete bottleneck；
- 引入 RALM；
- 把历史 acoustic embedding `E_<i` 显式反馈给 RALM；
- LocDiT 做 continuous local rendering；
- end-to-end joint training。

因此现有消融能证明的是：

1. **pure continuous 并不稳定**：去掉 FSQ 会变差；
2. **hierarchical residual modeling 很重要**：去掉 RALM 会变差；
3. **continuous acoustic history 很重要**：去掉 `E_<i` 会显著变差；
4. 但不能严格证明“tokenizer-free alone”贡献了多少。

所以更合理的收益归因是：

```text
稳定性
  ↑
FSQ semi-discrete bottleneck
  └── 把 semantic/prosody planning 与细声学渲染部分解耦

真实感 / 克隆能力
  ↑
continuous E_<i> + RALM
  └── 保留 speaker / timbre / recording condition / micro-prosody

最终波形质量
  ↑
LocDiT + AudioVAE
  └── 在连续 latent 上做局部高保真渲染
```

---

## 8. 和 StepAudio 3 Gen 的关系

这两个模型表面上路线完全相反：

- VoxCPM：tokenizer-free / continuous latent；
- StepAudio 3 Gen：16-layer RVQ / fully discrete generation。

但从结构思想上看，其实非常接近：

### StepAudio 3 Gen

```text
c0
│
└── semantic / temporal / coarse planning

c1 ... c15
│
└── residual acoustic detail
```

### VoxCPM

```text
TSLM + FSQ
│
└── semantic / prosody planning

continuous E_<i> + RALM
│
└── residual acoustic detail
```

共同思想是：

\[
\boxed{\text{coarse semantic path} + \text{fine acoustic residual path}}
\]

真正的差别在 fine acoustic path：

| | StepAudio 3 Gen | VoxCPM |
|---|---|---|
| 高层表示 | discrete `c0` | TSLM hidden + FSQ |
| 细节表示 | discrete RVQ `c1...c15` | continuous acoustic residual |
| 历史声学信息 | 完整 RVQ frame 经 adaptor 回灌 LLM | continuous latent 经 LocEnc 得到 `E_<i` 回灌 |
| 低层生成 | causal RVQ code predictor | LocDiT |
| waveform reconstruction | codec/Vocos | AudioVAE decoder |

所以更本质的问题不是“discrete vs continuous 谁一定更好”，而是：

> **怎样让 high-level LM 专注长程语义与韵律，同时又给低层声学细节保留一条足够高带宽的信息通路。**

VoxCPM 用 continuous residual path 解决；StepAudio 3 Gen 用多层 RVQ residual codes 解决。

---

## 9. 我对 VoxCPM 的核心理解

如果只记三件事：

1. **Tokenizer-free ≠ pure continuous**：VoxCPM 内部依然用 FSQ 给高层 representation 加 semi-discrete inductive bias。
2. **`E_<i` 很关键**：历史连续 acoustic embedding 被显式送回 RALM，论文消融显示去掉这条路径会大幅退化。
3. **真正的创新是分工**：TSLM/FSQ 管 semantic-prosody，RALM 管 acoustic residual，LocDiT 管 local continuous rendering。

因此比“它没有 tokenizer”更准确的一句话是：

> **VoxCPM 不再把语音压成一个外部离散 token 序列，而是在端到端模型内部，用 semi-discrete 高层规划 + continuous 声学残差保留语音信息。**

---

## 10. 参考资料

- VoxCPM Technical Report: https://arxiv.org/abs/2509.24650
- VoxCPM 官方 Demo / Ablation: https://voxcpm.github.io/VoxCPM-demopage/
- VoxCPM 官方 GitHub: https://github.com/OpenBMB/VoxCPM
- StepAudio 3 Gen 笔记：[`stepaudio-3-gen.md`](stepaudio-3-gen.md)

> 注：本文主要讨论 2025 年 VoxCPM / VoxCPM-0.5B 论文中的原始架构与消融，不等同于 2026 年发布的 VoxCPM2 全部新增能力。