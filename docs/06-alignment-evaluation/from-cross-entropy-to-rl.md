# 从 Cross Entropy 到 LLM 强化学习：没有 Ground Truth Token 时如何反向传播

> 目标：从最熟悉的 next-token prediction 出发，逐步理解 Policy Gradient、PPO、GRPO，以及为什么 RL 没有 ground-truth next token 仍然可以正常 `backward()`。

---

## 1. 一句话先抓住核心

SFT / Pretraining 的监督信号是：

> **“这个位置正确的 token 是什么。”**

RL 的监督信号则变成：

> **“模型刚才自己采样出来的这一串 action/token，整体应该更容易出现还是更不容易出现。”**

因此二者底层都在操作同一个量：

\[
\log \pi_\theta(y_t\mid s_t)
\]

区别只是：

- Cross Entropy 用 ground-truth token 当 target；
- Policy Gradient 用模型自己 sample 出来的 token，当作被 reward / advantage 加权的 action。

---

# 2. LLM 的 Cross Entropy 到底在做什么

假设输入：

```text
1 + 1 =
```

正确下一个 token 是 `2`。

模型输出 logits：

```text
2: 2.0
3: 1.0
4: 0.0
...
```

softmax 后得到：

\[
P(2)=0.665,\quad P(3)=0.245,\quad P(4)=0.090
\]

对于 target token `2`：

\[
L_{CE}=-\log P_\theta(2)
\]

对整段序列：

\[
L_{SFT}=-\sum_{t=1}^{T}\log \pi_\theta(y_t^*\mid x,y_{<t}^*)
\]

其中 \(y_t^*\) 是数据集提供的 ground-truth token。

### 直觉

Cross Entropy 实际就是：

> 找到 target token 的概率，然后取 negative log probability。

因此：

```text
模型给 GT token 的概率越高
        ↓
-log p 越小
        ↓
loss 越小
```

---

# 3. `get_logprob(logits, tokens)` 和 Cross Entropy 的关系

RL 代码里经常出现：

```python
log_probs_all = torch.log_softmax(logits, dim=-1)

token_log_probs = log_probs_all.gather(
    dim=-1,
    index=generated_tokens.unsqueeze(-1),
).squeeze(-1)
```

它做的事情只是：

```text
模型输出整个 vocab 的 log probability
                ↓
找到实际生成的 token
                ↓
取出 log P(token)
```

数学上：

\[
\mathrm{CE}(\mathrm{logits},y)=-\log P(y)
\]

所以：

```text
get_logprob = log P(y)
Cross Entropy = -log P(y)
```

本质上是同一个概率量。

RL 之所以更喜欢保留 `log_prob`，是因为后面还需要计算 advantage、importance sampling ratio、PPO clip 等。

---

# 4. RL 没有 Ground Truth Token，Loss 从哪里来？

假设模型自己生成：

```text
Q: 17 × 23 = ?

17×20=340
17×3=51
所以答案是 391
```

环境或 verifier 最后给：

\[
R=1
\]

这里没有任何人告诉模型：

```text
第 1 个 token 应该是什么
第 2 个 token 应该是什么
...
```

但是模型知道自己每一步实际采样 token 的概率：

\[
\pi_\theta(y_t\mid s_t)
\]

于是最基础的 Policy Gradient loss 可以写成：

\[
\boxed{
L_{PG}=-A\sum_t\log \pi_\theta(y_t\mid s_t)
}
\]

其中 \(A\) 是 Advantage。

可以先把它理解成：

> 这条 trajectory 比基准水平好多少。

---

# 5. Advantage 为正时发生什么

假设：

\[
A=+1
\]

则：

\[
L=-\sum_t\log\pi_\theta(y_t)
\]

这和 SFT 的 Cross Entropy 形式非常像。

梯度下降会让：

```text
这条成功 trajectory 中实际生成的 token
                     ↓
                 概率上升
```

因此 RL 相当于说：

> “你刚才这条路走得不错，以后更容易走出类似的路。”

---

# 6. Advantage 为负时发生什么

假设模型生成了一个失败 trajectory：

\[
A=-1
\]

那么：

\[
L=-(-1)\log p(y)=\log p(y)
\]

梯度下降会使 \(\log p(y)\) 更小，也就是：

\[
p(y)\downarrow
\]

所以：

```text
好 trajectory  → probability ↑
坏 trajectory  → probability ↓
```

这就是最基本的 Policy Gradient。

---

# 7. 为什么 RL 的 Loss 可以是负数？

Cross Entropy：

\[
L_{CE}=-\log p(y)\ge 0
\]

因为 \(0<p\le1\)。

但 RL：

\[
L_{RL}=-A\log p(y)
\]

如果 \(A<0\)，那么 loss contribution 可以小于 0。

例如：

\[
p=0.2,\quad \log p=-1.609
\]

若：

\[
A=-1
\]

则：

\[
L=-(-1)(-1.609)=-1.609
\]

完全正常。

优化器真正关心的是：

\[
\frac{\partial L}{\partial \theta}
\]

而不是 loss 是否为正。

> **RL 中 loss 的绝对数值通常没有 Cross Entropy 那么直观；梯度方向更重要。**

---

# 8. Reward 本身为什么不需要可微？

这是理解 RL 最关键的一步。

比如 reward 来自：

```python
reward = run_pytest(repo)
```

或者：

```text
Browser → 打开网页 → 点击 → 检查结果
```

甚至：

```text
另一个 LLM grader → 评分 0.8
```

这些过程都不可微。

但是 RL 不需要：

\[
\frac{\partial R}{\partial\theta}
\]

它只把 reward 当作权重：

\[
L=-R\log\pi_\theta(y)
\]

真正求导的是：

\[
\frac{\partial\log\pi_\theta(y)}{\partial\theta}
\]

数据流：

```text
environment / grader
        ↓
      reward
   （只是数字）
        ↓
reward / advantage × log probability
        ↓
Transformer forward
        ↓
loss.backward()
        ↓
optimizer.step()
```

---

# 9. Policy Gradient 的数学来源

RL 想最大化：

\[
J(\theta)=\mathbb E_{y\sim\pi_\theta}[R(y)]
\]

直接对 reward 求导通常不可行。

利用 log-derivative trick：

\[
\nabla_\theta \pi_\theta(y)
=\pi_\theta(y)\nabla_\theta\log\pi_\theta(y)
\]

可以得到：

\[
\nabla_\theta J
=
\mathbb E_{y\sim\pi_\theta}
\left[
R(y)\nabla_\theta\log\pi_\theta(y)
\right]
\]

而一条自回归序列：

\[
\log\pi_\theta(y)
=
\sum_t\log\pi_\theta(y_t\mid y_{<t})
\]

因此：

\[
\nabla_\theta J
=
\mathbb E
\left[
R
\sum_t
\nabla_\theta\log\pi_\theta(y_t\mid y_{<t})
\right]
\]

实现时取负号变成 minimization objective：

\[
L=-R\sum_t\log\pi_\theta(y_t)
\]

实际训练一般把 \(R\) 替换为方差更小、更稳定的 Advantage \(A\)。

---

# 10. 为什么需要 Advantage，而不是直接用 Reward？

如果所有 reward 都直接乘 log probability，梯度方差会很大。

最简单的 baseline：

\[
A=R-b
\]

例如：

```text
reward = 0.8
baseline = 0.5
advantage = +0.3
```

表示：

> 不只是“这次得了 0.8 分”，而是“比正常水平高 0.3”。

如果：

```text
reward = 0.4
baseline = 0.5
advantage = -0.1
```

则意味着应该轻微降低这条 trajectory 的概率。

---

# 11. GRPO：同一个 Prompt 做多次 Rollout

假设同一个 prompt 生成 4 条 trajectory：

```text
trajectory 1 → reward = 1
trajectory 2 → reward = 0
trajectory 3 → reward = 1
trajectory 4 → reward = 0
```

组内平均：

\[
\bar R=0.5
\]

简单写成：

\[
A_i=R_i-\bar R
\]

得到：

```text
trajectory 1: +0.5
trajectory 2: -0.5
trajectory 3: +0.5
trajectory 4: -0.5
```

标准化后常写成：

\[
A_i=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\epsilon}
\]

然后：

```text
成功轨迹的 token 概率 ↑
失败轨迹的 token 概率 ↓
```

这就是 Group Relative 的核心直觉。

---

# 12. 最大问题：Credit Assignment

假设 Agent trajectory 有 50 个 turns：

```text
turn 1   inspect files
turn 2   search
turn 3   edit
turn 4   test → fail
turn 5   debug
...
turn 48  找到真正原因
turn 49  修复
turn 50  test → pass
```

最后：

\[
R=1
\]

最朴素做法会让整个 trajectory 共用一个 advantage：

\[
A_1=A_2=\cdots=A_T
\]

但这意味着：

> 前面大量错误探索，也会跟最后真正正确的步骤一起被强化。

这就是 **credit assignment problem**：

> 最终 reward 到底应该归功于哪些 token、哪些 action、哪些 turn？

长 Agent RL 中，这比普通数学题 RL 难得多。

---

# 13. 为什么 PPO / GRPO 还要计算 Ratio

trajectory 往往由旧 policy \(\pi_{old}\) 生成，而 trainer 更新时使用的是新 policy \(\pi_\theta\)。

因此计算：

\[
r_t(\theta)
=
\frac{\pi_\theta(y_t\mid s_t)}
{\pi_{old}(y_t\mid s_t)}
\]

实践中：

\[
r_t=
\exp(
\log\pi_\theta-
\log\pi_{old}
)
\]

PPO-style loss：

\[
L=-\min\left(
 r_tA_t,
 \operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t
\right)
\]

### 直觉

如果某个好 token：

```text
old probability = 0.10
new probability = 0.20
```

说明已经强化很多。

PPO clip 的作用就是：

> “方向对，但一次不要走太远。”

避免 policy 在一次 optimizer step 中发生过大的分布漂移。

---

# 14. SFT 与 RL 的完整对照

| 项目 | SFT / Pretraining | RL / Policy Gradient |
|---|---|---|
| token 来源 | Dataset ground truth | Policy 自己 sample |
| 基础量 | \(-\log p(y^*)\) | \(\log p(y_{sample})\) |
| 权重 | 通常 1 / mask | Reward / Advantage |
| 好行为 | 提高 GT token 概率 | Advantage > 0 时提高 |
| 坏行为 | 不在 GT 中 | Advantage < 0 时降低 |
| reward 是否需可微 | 不涉及 | 不需要 |
| loss 是否必须 ≥ 0 | 通常是 | 不需要 |
| 核心难题 | 正确标签质量 | exploration + reward + credit assignment |

---

# 15. 一段极简 PyTorch 伪代码

## SFT

```python
logits = model(input_ids)

loss = F.cross_entropy(
    logits[:, :-1],
    labels[:, 1:],
)

loss.backward()
optimizer.step()
```

## 最简单 Policy Gradient

```python
# rollout 阶段：模型自己生成
with torch.no_grad():
    generated_tokens = policy.generate(prompt)

# environment / verifier 给 reward
reward = grader(generated_tokens)
advantage = reward - baseline

# trainer 再 forward 一次，得到当前 policy 对这些 action 的概率
logits = policy(prompt + generated_tokens)
log_probs = gather_generated_token_log_probs(
    logits,
    generated_tokens,
)

loss = -(advantage * log_probs).mean()

loss.backward()
optimizer.step()
```

真实 PPO / GRPO 会进一步加上：

- old log probability；
- importance sampling ratio；
- clipping；
- KL regularization；
- token mask；
- group-normalized advantage；
- distributed rollout / trainer synchronization。

---

# 16. 最重要的心智模型

不要把 RL 理解成“没有 label，因此无法算 loss”。

应该理解成：

```text
SFT
正确 token
   ↓
-log P(correct token)
   ↓
backward

RL
模型自己 sample 的 token
   ↓
reward / advantage
   ↓
-A × log P(sampled token)
   ↓
backward
```

两者最终都通过：

\[
\log \pi_\theta(y_t\mid s_t)
\]

把梯度传回 Transformer。

最大的变化不是“backward 方法变了”，而是：

> **训练信号从 token-level ground truth，变成了 trajectory-level / action-level reward。**

---

## 17. 下一步学习

理解这篇之后，可以继续看：

1. REINFORCE 为什么无偏但高方差；
2. Value Model / Critic 如何估计 Advantage；
3. PPO 的 clipping 为什么有效；
4. GRPO 为什么不需要单独训练 Critic；
5. DAPO / Dynamic Sampling 为什么过滤全对与全错 group；
6. Agent RL 的 turn-level / token-level credit assignment；
7. Fully Async RL 中 policy staleness 与 importance sampling。
