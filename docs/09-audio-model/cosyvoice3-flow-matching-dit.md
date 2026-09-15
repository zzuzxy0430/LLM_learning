# CosyVoice 3 中 Flow Matching 与 DiT 的原理

> 说明：当前仓库中，使用 DiT 作为 Flow Matching 速度场估计器的是 **CosyVoice 3**。CosyVoice 2 使用的是卷积/U-Net 风格的 `CausalConditionalDecoder`。DiT 并不是另一套 Flow Matching 算法，而是 Flow Matching 中负责预测速度场的骨干网络。

## 1. 整体生成链路

```text
文本
  │
  ▼
LLM 预测 25 Hz 离散 speech token
  │
  │ token embedding，并上采样到 50 Hz
  ▼
Conditional Flow Matching
  ├── 初始状态：高斯噪声 mel
  ├── 条件：speech token、prompt mel、speaker embedding
  └── DiT：预测当前状态的速度
  │
  │ 10 步 ODE 积分
  ▼
80 维、50 Hz mel 频谱
  │
  ▼
HiFT vocoder
  │
  ▼
24 kHz 音频波形
```

LLM 主要负责离散层面的内容、发音和粗粒度韵律；Flow Matching 负责把这些离散信息还原成连续、细腻的声学特征；HiFT 再把 mel 频谱转换为最终波形。

## 2. Flow Matching 学习的目标

设：

- $x_1$：训练数据中的真实 80 维 mel 频谱；
- $z\sim\mathcal N(0,I)$：与 mel 形状相同的高斯噪声；
- $t\sim U(0,1)$：随机采样的时间；
- $\sigma_{\min}=10^{-6}$。

CosyVoice 构造一条从噪声到真实 mel 的近似直线路径：

$$
x_t=\left[1-(1-\sigma_{\min})t\right]z+t x_1.
$$

因此：

- 当 $t=0$ 时，$x_t=z$，输入完全是噪声；
- 当 $t=1$ 时，$x_t=x_1+\sigma_{\min}z$，几乎就是真实 mel。

这条路径对时间求导，得到训练目标速度：

$$
u_t=\frac{dx_t}{dt}=x_1-(1-\sigma_{\min})z.
$$

DiT 接收当前状态和所有条件：

$$
(x_t,\ t,\ \mu,\ \mathrm{prompt},\ \mathrm{speaker}),
$$

并预测速度：

$$
v_\theta(x_t,t,c)\approx u_t.
$$

训练损失是有效 mel 区域上的均方误差：

$$
\mathcal L_{\mathrm{CFM}}
=\mathbb E\left[\left\|M\odot\left(v_\theta(x_t,t,c)-u_t\right)\right\|_2^2\right],
$$

其中 $M$ 是用于排除 padding 的 mask。

直观上，可以把训练理解成：

> 随机取“噪声到真实语音”路径上的一个位置，然后告诉模型：站在这里时，应该朝哪个方向、以多快的速度移动。

模型不会直接看到终点 $x_1$ 和噪声 $z$，只能根据 $x_t$、时间和语音条件预测方向。MSE 的最优解对应给定这些观测时的条件平均速度，该速度场可以把噪声分布连续地运输到条件语音分布。

相关实现：

- 构造 $x_t$：[flow_matching.py 第 181 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L181)
- 构造目标速度 $u_t$：[flow_matching.py 第 182 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L182)
- DiT 预测与 MSE：[flow_matching.py 第 191 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L191)

## 3. DiT 在 CosyVoice 3 中的作用

DiT 原名 Diffusion Transformer。它虽然起源于扩散模型，但本质上是一个带时间条件的 Transformer，因此也可以用于预测 Flow Matching 的速度场。

CosyVoice 3 中 DiT 的主要输入如下：

| 输入 | 维度 | 含义 |
|---|---:|---|
| $x_t$ | 80 | 当前时刻的带噪 mel |
| `mu` | 80 | 由 LLM speech token 得到的连续条件 |
| `cond` | 80 | prompt/reference mel；非 prompt 区域为零 |
| `spks` | 80 | 从 192 维 x-vector 投影得到的说话人条件 |
| $t$ | 标量 | 当前 Flow/ODE 时间 |

这些输入并不是以相同方式进入 DiT：

- $x_t$ 是 ODE 当前状态，也是 DiT 要处理的主体，严格来说不属于条件；
- `mu`、`cond` 和 `spks` 通过输入层的逐帧特征拼接进入 DiT；
- 时间 $t$ 不参与输入特征拼接，而是通过时间嵌入和 AdaLayerNorm 调制每一层。

这里的 `mu` 容易被误解：在 CosyVoice 3 代码里，它是经过嵌入、轻量前瞻处理并上采样后的 **speech-token 条件**，不是待生成 mel 的均值。

### 3.1 `mu`、prompt 和 speaker：逐帧特征拼接

进入 `InputEmbedding` 前，$x_t$、`mu` 和 `cond` 都从 `[B, 80, T]` 转换为 `[B, T, 80]`。全局 speaker embedding 则沿时间维复制 $T$ 次。因此，对第 $i$ 个 mel 时间帧，输入为：

$$
q_i=[x_{t,i};\ \mathrm{cond}_i;\ \mu_i;\ e_{\mathrm{spk}}].
$$

每一项都是 80 维，所以：

$$
q_i\in\mathbb R^{80+80+80+80}=\mathbb R^{320}.
$$

随后，每个 $q_i$ 被线性投影为一个 1024 维的 DiT token：

$$
h_i=W_{\mathrm{in}}q_i+b.
$$

这里的拼接发生在最后一个 feature/channel 维度，**不会增加序列长度**。它不是下面这种 prefix-token 方式：

```text
[prompt tokens] [speaker token] [mu tokens] [audio tokens]
```

实际结构是：

```text
时间帧 1: [x_t,1 | prompt_1 | mu_1 | speaker] → DiT token 1
时间帧 2: [x_t,2 | prompt_2 | mu_2 | speaker] → DiT token 2
时间帧 3: [x_t,3 | prompt_3 | mu_3 | speaker] → DiT token 3
...
```

其中：

- `mu` 是离散 speech token 经 embedding、前瞻层处理并由 25 Hz 重复上采样至 50 Hz 后的逐帧条件；
- `cond` 是与 mel 等长的 prompt 条件，prompt 区域放入真实参考 mel，其他区域填零；
- `spks` 是全局说话人向量，但在拼接前会复制到每个时间帧。

#### `mu` 与 mel 的长度关系

原始 speech token 和 mel 的时间长度确实是 $1:2$。假设原始 speech token 长度为 $N$：

| 表征 | 帧率 | 长度 |
|---|---:|---:|
| 原始 speech token / 前瞻处理后的 token 表征 | 25 Hz | $N$ |
| 上采样后的 `mu` | 50 Hz | $2N$ |
| mel | 50 Hz | $T_{\mathrm{mel}}=2N$ |

代码没有使用连续插值，而是用 `repeat_interleave(2)` 将每个 token 表征连续复制两次：

```text
上采样前：[mu_0, mu_1, mu_2, ...]
上采样后：[mu_0, mu_0, mu_1, mu_1, mu_2, mu_2, ...]
```

对应代码为：

```python
h = self.pre_lookahead_layer(token)
h = h.repeat_interleave(self.token_mel_ratio, dim=1)  # token_mel_ratio = 2
```

因此，应该区分两个阶段：

- **进入 Flow 模型之前**：原始 speech token 与 mel 的长度比为 $1:2$；
- **进入 DiT 时**：上采样后的 `mu` 已经与 mel 对齐，二者长度比为 $1:1$。

若用 $\widetilde\mu_j$ 表示上采样前第 $j$ 个 token 表征，则进入第 $i$ 个 mel 位置的条件是：

$$
\mu_i=\widetilde\mu_{\lfloor i/2\rfloor}.
$$

于是第 $i$ 个 DiT 输入 token 可以更明确地写为：

$$
h_i=W_{\mathrm{in}}
\left[x_{t,i};\ \mathrm{cond}_i;\
\widetilde\mu_{\lfloor i/2\rfloor};\ e_{\mathrm{spk}}\right]+b.
$$

`speaker` 的处理与 `mu` 不同：它没有原始时间序列，而是将同一个全局说话人向量从 `[B, 80]` 复制为 `[B, T_mel, 80]`。最终进入 DiT 时的形状为：

| 输入 | 进入 DiT 时的形状 |
|---|---|
| $x_t$ | `[B, T_mel, 80]` |
| `mu` | `[B, T_mel, 80]` |
| `cond` | `[B, T_mel, 80]` |
| `speaker` | `[B, 80] -> [B, T_mel, 80]` |

另外，`pre_lookahead_len=3` 表示每个 token 位置可以利用有限的未来上下文，不会把 `mu` 的长度扩大三倍。长度扩大两倍只来自 `token_mel_ratio=2`。相关实现见 [flow.py 第 345 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L345)。

投影后的 DiT token 还会叠加因果卷积位置特征。代码见 [dit.py 第 76 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py#L76)。

默认 CosyVoice 3 配置中的 DiT 包含：

- 隐藏维度：1024；
- Transformer 层数：22；
- Attention heads：16；
- 每个 head 的维度：64；
- FFN 扩展倍数：2；
- 输出维度：80，即每个 mel bin 的速度。

配置见 [cosyvoice3.yaml 第 64 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml#L64)。

当前配置实际使用普通 `DiTBlock`。条件采用输入端拼接的 early fusion，而不是独立的 cross-attention。源码虽然还实现了 `MMDiTBlock`，但该配置没有启用它。

### 3.2 Prompt mel 如何注入 DiT

Prompt mel 也沿特征维注入 DiT，但它只在完整序列的前缀位置非零。它既不是 AdaLayerNorm 条件，也不是额外插入到序列前面的 prefix token。

#### 3.2.1 从 prompt 音频提取三类条件

前端首先把 24 kHz prompt 音频转换成 80 维声学特征：

$$
P\in\mathbb R^{B\times T_p\times80},
$$

其中 $T_p$ 是 prompt mel 的帧数。实现见 [frontend.py 第 130 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/cli/frontend.py#L130)。

同一段 prompt 音频还会产生另外两类信息：

| Prompt 信息 | 在 Flow 中的作用 |
|---|---|
| prompt speech token | 提供与参考语音对应的离散内容和粗粒度声学信息，并成为 `mu` 的前缀 |
| prompt mel | 提供连续、细粒度的参考声学信息，并成为 `cond` 的前缀 |
| speaker embedding | 提供全局说话人信息，并复制到所有 mel 位置 |

#### 3.2.2 构造与完整 mel 等长的 `cond`

推理时先把 prompt speech token 与待生成的 speech token 沿时间维连接：

```python
token = torch.concat([prompt_token, token], dim=1)
```

它们经过 embedding、前瞻层和二倍重复上采样后，形成包含 prompt 与目标区域的完整 `mu`。设完整 mel 长度为：

$$
T=T_p+T_g,
$$

其中 $T_g$ 是需要生成的 mel 长度。代码随后创建一个与完整 mel 等长的全零条件矩阵：

$$
\mathrm{cond}\in\mathbb R^{B\times T\times80},
$$

并只在前 $T_p$ 个位置写入真实 prompt mel：

```python
conds = torch.zeros([1, mel_len1 + mel_len2, 80])
conds[:, :mel_len1] = prompt_feat
```

因此：

$$
\mathrm{cond}_i=
\begin{cases}
P_i,& i<T_p,\\
0,& i\ge T_p.
\end{cases}
$$

对应的时间布局是：

```text
时间位置：  0 ........ T_prompt-1 | T_prompt ........ T_total-1
cond：     [   prompt mel         |          0                ]
mu：       [ prompt speech token  | generated speech token    ]
```

实现见 [flow.py 第 385 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L385) 和 [flow.py 第 398 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L398)。

#### 3.2.3 在 DiT 输入层逐帧拼接

`cond` 进入 DiT 后与 $x_t$、`mu` 和 speaker embedding 沿最后一个 feature/channel 维度拼接：

$$
h_i^0=W_{\mathrm{in}}
\left[x_{t,i};\ \mathrm{cond}_i;\ \mu_i;\ e_{\mathrm{spk}}\right]+b.
$$

因此，prompt 区域和目标生成区域的输入分别为：

```text
Prompt 区域：
[x_t,i | prompt_mel_i | prompt_mu_i | speaker]

生成区域：
[x_t,i |      0       | target_mu_i | speaker]
```

这不会增加 DiT 的序列长度。实现见 [dit.py 第 145 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py#L145) 和 [dit.py 第 91 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py#L91)。

#### 3.2.4 Prompt 信息如何影响后续生成帧

目标生成区域的 `cond_i` 虽然为零，但 prompt 前缀位置的隐藏状态已经直接接收了 prompt mel。随后，Transformer self-attention 把这些信息传播到后面的生成位置：

```text
prompt mel
   │ 直接注入前缀位置
   ▼
prompt hidden states
   │ self-attention / causal chunk attention
   ▼
后续生成位置
```

- 离线模式使用完整 attention，生成位置可以利用 prompt 前缀；
- 流式模式使用 chunk attention，后续 chunk 可以看到 prompt 与历史 chunk，但不能看到未来 chunk；
- 因果卷积位置编码也会沿时间方向把局部前缀信息传给后续位置。

所以 prompt mel 不是在每个目标位置重复一份，而是先注入前缀位置，再通过序列建模影响后续 mel，帮助延续参考语音的音色、录音环境和声学风格。

#### 3.2.5 它是软条件，不是硬钳制

CosyVoice 3 的 `CausalConditionalCFM` 没有在每个 ODE 步骤中执行下面这种操作：

```python
x_t[:, :, :prompt_len] = prompt_mel
```

换言之，prompt 区域的 ODE 状态没有被强制替换成真实 prompt mel。完整序列仍然从噪声开始：

```text
x_t：  [noise prompt part | noise target part]
cond： [real prompt mel   | zeros]
```

训练时，提供 prompt 的样本在前缀区域满足 `cond = x1 = prompt mel`，而损失覆盖完整有效序列。因此模型会学习让最终生成的 prompt 区域接近真实 prompt：

$$
x(1)=[\widehat P;\widehat Y],\qquad \widehat P\approx P.
$$

但这里不是严格的 $\widehat P=P$：速度场预测误差、10 步 Euler 积分误差以及 CFG 都可能造成差异，代码也没有在最终输出上执行逐元素覆盖。

DiT 在每个 ODE 步骤都把 prompt mel 当作不随时间变化的条件通道，并生成包含 prompt 区域和目标区域的完整 mel。积分结束后，代码裁掉前 $T_p$ 个生成帧，只把新生成的目标区域交给 vocoder：

```python
feat = feat[:, :, mel_len1:]
```

实现见 [flow.py 第 412 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L412)。因此，更准确的描述是 **prefix acoustic conditioning（前缀声学条件）**，而不是 hard inpainting/clamping（硬性补全或钳制）。

#### 3.2.6 训练时如何学习 prompt conditioning

训练时，代码从真实 mel 的开头随机选取一段作为 `cond`：

```python
conds = torch.zeros(feat.shape)
index = random.randint(0, int(0.3 * feat_len))
conds[i, :index] = feat[i, :index]
```

具体策略为：

- 大约一半训练样本不提供 prompt mel；
- 其余样本从整段语音的前 0～30% 随机选取真实 mel 前缀；
- Flow Matching 仍然以整段真实 mel 为目标，模型由此学会利用声学前缀继续生成；
- CFM 训练中的 20% classifier-free condition dropout 还可能同时清零 `mu`、`cond` 和 speaker，以便推理时计算有条件与无条件速度。

训练前缀构造见 [flow.py 第 350 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L350)，CFG 条件丢弃见 [flow_matching.py 第 184 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L184)。

Prompt mel 的完整注入过程可以概括为：

$$
\boxed{
\text{prompt mel}
\rightarrow
\text{写入 cond 的前缀位置}
\rightarrow
\text{与 }x_t,\mu,\text{speaker 逐帧拼接}
\rightarrow
\text{通过 self-attention 影响后续生成}
}
$$

### 3.3 时间 $t$：逐层调制

与上述逐帧特征不同，标量时间 $t$ 不会被拼入 $q_i$，也不会形成一个时间 prefix token。它经过正弦位置编码和 MLP，生成一个全局时间向量，再送入每一个 DiT Block 的 AdaLayerNorm，并在最终输出归一化层中再次使用。下一节给出具体过程。

## 4. 时间条件如何控制 DiT

标量时间 $t$ 首先经过：

```text
正弦时间编码 → MLP → 1024 维时间向量
```

每一个 DiT Block 都通过 AdaLayerNorm 使用该时间向量。简化后可写为：

$$
\widehat h=\operatorname{LN}(h)\odot(1+s(t))+b(t).
$$

时间向量还产生 attention 和 FFN 的门控值：

$$
h\leftarrow h+g_{\mathrm{attn}}(t)\operatorname{Attention}(\widehat h),
$$

$$
h\leftarrow h+g_{\mathrm{ffn}}(t)\operatorname{FFN}(\widehat h).
$$

所以同一个 Transformer 在不同时间点会执行不同的变换：

- $t$ 较小时，输入更接近噪声，网络主要建立粗粒度语音结构；
- $t$ 较大时，输入已接近 mel，网络更多地修正局部细节；
- speech token 约束内容和粗粒度韵律；
- speaker embedding 与 prompt mel 约束说话人音色和参考声学信息。

DiT Block 实现见 [modules.py 第 500 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/DiT/modules.py#L500)。

## 5. 推理：从噪声积分到 mel

推理从高斯噪声开始：

$$
x(0)=z,
$$

随后求解常微分方程：

$$
\frac{dx}{dt}=v_\theta(x,t,c).
$$

CosyVoice 当前实现使用 10 步 Euler 积分：

$$
x_{k+1}=x_k+(t_{k+1}-t_k)v_\theta(x_k,t_k,c).
$$

积分时间点经过 cosine 调度。完成 10 步后，$x(1)$ 就是生成的 mel 频谱，随后交给 HiFT vocoder。

相关实现：

- Euler 积分：[flow_matching.py 第 101 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L101)
- 10 步推理设置：[flow.py 第 404 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L404)

它与传统 DDPM 采样的关键区别是：

- DiT 输出的是速度 $dx/dt$，而不是 DDPM 中常见的噪声 $\epsilon$；
- 推理求解确定性 ODE，不会在每一步重新加入随机噪声；
- 训练时可以直接采样任意 $t$，无需逐步执行前向加噪；
- 由于条件路径较直接，少量 ODE 步数也能获得较好的结果。

当前 `CausalConditionalCFM` 使用一张预先生成的固定噪声表，而不是每次推理重新采样噪声。这样可以让流式分块推理和整段推理在相同位置使用一致的初始噪声。实现见 [flow_matching.py 第 196 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L196)。

## 6. Classifier-Free Guidance

训练时，以 20% 的概率同时清零 speech-token、speaker 和 prompt 条件，使同一个 DiT 同时学习：

- $v_{\mathrm{cond}}$：有条件速度；
- $v_{\mathrm{uncond}}$：无条件速度。

推理时二者组合为：

$$
v_{\mathrm{CFG}}
=(1+w)v_{\mathrm{cond}}-w v_{\mathrm{uncond}}.
$$

仓库默认 $w=0.7$，因此：

$$
v_{\mathrm{CFG}}=1.7v_{\mathrm{cond}}-0.7v_{\mathrm{uncond}}.
$$

这会增强生成结果对 speech token、说话人音色和 prompt 的服从程度。训练条件丢弃见 [flow_matching.py 第 184 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L184)，推理 CFG 组合见 [flow_matching.py 第 117 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow_matching.py#L117)。

## 7. 流式生成

CosyVoice 3 的 DiT 同时支持离线和流式推理：

- 离线模式使用完整时间上下文；
- 流式模式使用 chunk attention mask，不能看到后续 chunk；
- 配置中的 25 个 speech token 会扩展成 50 个 mel frame，即大约 1 秒的声学 chunk；
- 输入侧允许 3 个 speech token 的有限前瞻；
- 因果卷积位置编码不会读取未来帧。

训练时随机在流式与非流式模式之间切换，使同一模型同时适配两种推理方式。实现见 [flow.py 第 334 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/flow.py#L334) 和 [dit.py 第 163 行](https://github.com/FunAudioLLM/CosyVoice/blob/main/cosyvoice/flow/DiT/dit.py#L163)。

## 8. 为什么使用 Flow Matching + DiT

1. **训练目标简单**：把生成建模转化为有监督的速度回归。
2. **采样步数较少**：近似直线路径配合 ODE，仓库默认只使用 10 步。
3. **长时建模能力强**：Transformer attention 适合捕获跨较长时间的韵律和声学关系。
4. **易于扩大模型规模**：DiT 可以通过增加宽度、层数和 attention heads 扩展容量。
5. **时间条件表达自然**：AdaLayerNorm 让每一层都根据当前噪声阶段动态工作。
6. **兼容流式生成**：通过因果卷积和 chunk attention mask 限制未来信息。
7. **条件控制统一**：speech token、prompt mel 和 speaker embedding 可以共同控制同一个速度场。

## 9. 一句话总结

> CosyVoice 3 先让 LLM 给出“要说什么”的离散 speech token，再让时间条件化的 DiT 学习一个速度场，用 10 步 ODE 把高斯噪声运输成符合 token、音色和 prompt 的连续 mel 频谱。
