# MiMo-V2.6 RL：大规模 Agentic RL 训练机制笔记

> 更新时间：2026-09-17  
> 说明：MiMo-V2.6 的 RL 训练仍在进行中，小米 MiMo 团队表示会在未来几周逐步开源更多细节。因此本文严格区分 **官方确认**、**可由公开指标直接推导**、以及 **合理推断**。

---

## 1. 当前官方确认了什么

MiMo 团队公开表示，MiMo-V2.6 正在扩展三件事：

1. **Compute**：每个 RL step 约 20 亿 token，`1,568 prompts × 16 rollouts`，Fully Async；
2. **Environments / Harnesses**：Multi-task Agentic RL，一次 run 中混合多个 harness；
3. **Grader Compute**：Agentic In-group Credit Assignment，同时使用 test-case reward 与 rubric-based reward。

公开来源：

- 实时页面：https://mimo.xiaomi.com/rl/
- MiMo-V2-Flash 技术报告：https://arxiv.org/abs/2601.02780
- MiMo-V2-Flash GitHub：https://github.com/XiaomiMiMo/MiMo-V2-Flash

媒体对 MiMo-V2.6 官方公开信息的同步报道：

- IT之家：https://www.ithome.com/1/003/555.htm
- 澎湃新闻：https://www.thepaper.cn/newsDetail_forward_34088356

---

# 2. `1568 prompts × 16 rollouts` 到底是什么意思

最重要的是：

> **1568 更合理地理解为一个 RL step 中的 prompt/group batch size，而不是最终只有 1568 条 trajectory 参加训练。**

同一个 prompt 会采样 16 次：

```text
Prompt A
  ├─ rollout 1
  ├─ rollout 2
  ├─ rollout 3
  ├─ ...
  └─ rollout 16
```

因此：

```math
1568\times16=25,088
```

即一个 step 约有 25K 条 trajectories。

如果每 step 约 2B token：

```math
\frac{2\times10^9}{25088}\approx79.7K
```

平均每条 trajectory 已经是数万到约 10 万 token 的量级。

这和长上下文、多轮 Agent 任务非常吻合。

---

# 3. 什么叫 Reward Distribution

假设同一个 Prompt A rollout 16 次。

最简单的 binary verifier：

```text
rollout 1  → reward = 1
rollout 2  → reward = 0
rollout 3  → reward = 0
rollout 4  → reward = 1
...
rollout 16 → reward = 0
```

得到：

```math
R_A=[1,0,0,1,\dots,0]
```

这就是该 prompt 在当前 policy 下 16 次尝试形成的 **经验 reward distribution**。

如果成功 5 次：

```math
passrate=\frac5{16}=0.3125
```

它不是一个额外神经网络输出的概率分布，而只是：

> **同一个 prompt 的多次 rollout 经过 grader 后得到的一组 reward。**

---

# 4. 为什么同一个 Prompt 要 Rollout 16 次

如果只做一次：

```text
Prompt A → fail
```

只能知道这一次失败。

做 16 次之后，可以估计当前 policy 对这道题的稳定程度：

### 情况 A：全失败

```text
0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0
```

```math
p\approx0
```

### 情况 B：全成功

```text
1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1
```

```math
p\approx1
```

### 情况 C：有时成功有时失败

```text
1 0 0 1 0 1 1 0
1 0 1 0 0 1 0 1
```

```math
p\approx0.5
```

第三种情况特别有 RL 学习价值，因为同一个初始任务下：

> 有成功 trajectory，也有失败 trajectory，可以做组内相对比较。

---

# 5. Dynamic Sampler 在做什么

这里要把 **Prompt Scheduler** 和 **Dynamic Sampling** 放在一起理解。

MiMo-V2-Flash 技术报告已经公开了 Data Scheduler 的一些设计：

- 参考 historical pass rate；
- 支持不同数据源的 sample quota；
- 支持 scheduling priority；
- sequence 完成 reward 后可以立刻继续调度；
- 使用 fine-grained sequence scheduling，而不是等待整个 micro-batch；
- 支持 partial rollout；
- 会考虑 GPU load balancing。

因此整体可以理解为：

```text
多个 dataset / harness
        ↓
quota / priority / historical passrate
        ↓
Data Scheduler
        ↓
candidate prompt
        ↓
rollout ×16
        ↓
grader
        ↓
reward distribution
        ↓
Dynamic Sampler
   ↙            ↘
reject         accept
                 ↓
         进入训练 group pool
```

---

# 6. Dynamic Sampling 为什么常过滤“全对 / 全错”组

以最简单的 GRPO 为例：

```math
A_i=
\frac{R_i-\mu_R}{\sigma_R+\epsilon}
```

如果 16 个 reward 全一样：

```math
R=[0,0,\dots,0]
```

或者：

```math
R=[1,1,\dots,1]
```

那么：

```math
\sigma_R=0
```

组内没有 relative signal。

因此动态采样通常倾向于继续补充新的 prompt，避免训练 batch 被大量 zero-variance groups 占满。

对于 binary reward：

```math
\mathrm{Var}(R)=p(1-p)
```

当：

```math
p=0.5
```

方差最大。

这意味着当前模型“有时会、有时不会”的题，天然提供更丰富的组内比较信号。

> 注意：这只是 Dynamic Sampling 的核心直觉。MiMo-V2.6 目前没有公开完整 acceptance rule，不能把它简单等同于 `std(reward) > 0`。

---

# 7. 一个完整的 Dynamic Sampling 例子

假设 trainer 最终需要：

```math
1568\text{ 个有效 prompt groups}
```

scheduler 先拿 2000 个候选 prompt。

每题 rollout 16 次：

```math
2000\times16=32000
```

条 trajectories。

例如：

```text
Prompt 001:
0000000000000000   passrate=0

Prompt 002:
1111111111111111   passrate=1

Prompt 003:
1001011001100101   passrate=0.50

Prompt 004:
0000001000000000   passrate=0.0625

Prompt 005:
1111111110111111   passrate=0.9375
```

如果其中只有 1400 个 group 达到当前 acceptance 条件：

```text
accepted = 1400
required = 1568
```

还差：

```math
1568-1400=168
```

scheduler 就继续异步补采新的 prompts。

直到凑够足够的 training groups。

这比：

```text
固定取 1568 prompts
→ 等全部跑完
→ 直接训练
```

更适合 Agent RL，因为不同 trajectory 的完成时间可能相差非常大。

---

# 8. 1568 个 Group 最后对应多少训练 Sequence

如果一个被接受的 group 保留完整的 16 个 rollouts：

```math
1568\times16=25,088
```

所以真正参与 policy update 的 sequence batch size 是约 25K，而不是 1568。

因此更合理的流程是：

```text
大量候选 prompts
        ↓
每个 prompt rollout ×16
        ↓
Dynamic Sampling 筛 group
        ↓
凑够约 1568 accepted groups
        ↓
1568 × 16 ≈ 25K trajectories
        ↓
credit assignment
        ↓
policy gradient / PPO-style update
```

而不是：

```text
25K trajectories
    ↓
只挑 1568 条 trajectory
    ↓
backward
```

---

# 9. Reward 不一定只有 0 / 1

MiMo-V2.6 已经明确提到两类 reward：

- test-case reward；
- rubric-based reward。

因此可能存在：

```text
trajectory #7
   │
   ├─ unit tests: 8 / 10
   ├─ rubric grader: 0.7
   └─ 其他规则指标
```

最终得到连续 reward：

```math
R_7=0.76
```

于是 16 条 trajectory 可能形成：

```math
R=[0.92,0.10,0.81,0.23,0.76,\dots]
```

目前官方还没有公开这些 reward 的完整加权公式。

---

# 10. Agentic In-group Credit Assignment 为什么重要

普通 GRPO 最简单的做法是：

> 一条 trajectory 整体得到一个 advantage，然后整条轨迹上的 token 共用这个 advantage。

例如：

```text
trajectory 7
turn 1
turn 2
turn 3
...
turn 50
```

如果：

```math
A_7=+0.8
```

最粗粒度做法等价于：

```text
turn 1  → +0.8
turn 2  → +0.8
...
turn 50 → +0.8
```

但是 Agent trajectory 往往是：

```text
turn 1   inspect
turn 2   猜错
turn 3   改错文件
turn 4   test fail
turn 5   debug
...
turn 48  找到真正原因
turn 49  修复
turn 50  test pass
```

最终成功，并不意味着前面的错误尝试都应该被同样强化。

因此 MiMo-V2.6 特意强调：

> **Agentic In-group Credit Assignment**

说明它正在解决一个更细的问题：

> **同一 group 内，哪些 trajectory 更好；一条长 trajectory 内，又是哪些 turn / action 真正导致了最终成功。**

截至 2026-09-17，官方还没有公开该算法的具体数学形式。

---

# 11. Fully Async RL 为什么必要

同步 RL：

```text
生成整个 batch
      ↓
等最慢 trajectory
      ↓
全部 reward
      ↓
trainer update
```

Agent 任务特别容易出现 long-tail：

```text
trajectory A → 10 turns → 3 min
trajectory B → 30 turns → 15 min
trajectory C → 80 turns → 45 min
```

如果必须等 C，A/B 所占资源就容易空转。

MiMo-V2-Flash 已公开 fine-grained sequence scheduling 和 partial rollout，MiMo-V2.6 又进一步明确称本轮训练是 **Fully Async**。

更合理的结构：

```text
Prompt Scheduler
     │
     ├─ rollout worker 1 ─→ reward
     ├─ rollout worker 2 ─→ reward
     ├─ rollout worker 3 ─→ reward
     └─ rollout worker N ─→ reward
                         ↓
                 training queue
                         ↓
                      trainer
```

不同 rollout 完成后可以立即进入后续处理，不需要等待整个 batch 对齐。

---

# 12. Async RL 的代价：Policy Staleness

Fully Async 会带来新问题。

某个 trajectory 开始时使用：

```math
\pi_{\theta_t}
```

它跑了很久。

等完成并准备训练时，trainer 可能已经更新到：

```math
\pi_{\theta_{t+k}}
```

于是：

```text
rollout policy ≠ current training policy
```

这就是 policy staleness / train-inference mismatch。

PPO / importance sampling 会计算：

```math
r_t=
\frac{\pi_{new}(a_t|s_t)}
{\pi_{old}(a_t|s_t)}
```

实践中：

```math
r_t=\exp(
\log\pi_{new}-\log\pi_{old}
)
```

如果两者差得太远，说明这条旧 trajectory 对当前 policy 已经不够 on-policy。

---

# 13. MiMo-V2-Flash 已公开的 Partial Rollout

MiMo-V2-Flash 技术报告已经明确：

- 支持 partial rollout；
- 超长 trajectory 可以跨 training step；
- 会限制 staleness；
- 会限制 partial samples 在 batch 中的比例；
- 使用 staleness-aware truncated importance sampling。

其目的就是：

> 不因为少数超长 Agent trajectory 阻塞整个 RL pipeline，同时又控制 off-policy 程度。

因此 MiMo-V2.6 的 Fully Async 很可能是在上一代这套基础设施上继续扩展。

这里属于 **合理推断**，不是 V2.6 已公布实现细节。

---

# 14. 为什么 MiMo 特别强调 Grader Compute

传统 RLVR：

```text
final answer == ground truth
        ↓
reward = 0 / 1
```

Code RL：

```text
pytest
 ↓
passed / failed
```

但 Agent trajectory 可能有 50+ turns：

```text
search
→ inspect files
→ edit
→ test
→ debug
→ browser
→ tool
→ retry
→ final
```

只看最终成败会导致非常粗糙的 credit assignment。

所以 MiMo 将 scaling 分成：

```text
1. rollout compute
2. environment / harness compute
3. grader compute
```

第三项尤其值得注意：

> grader 本身开始成为一个独立的 scaling axis。

更强的 verifier、rubric grader 和 step/turn-level assessment，可能直接决定 RL signal 的质量。

---

# 15. Multi-task Agentic RL 的含义

MiMo-V2.6 公开称：

> 多任务 Agentic RL，一次 run 中混合多个 harness。

这意味着训练不再一定是：

```text
Code RL
训练完
↓
Browser RL
训练完
↓
General reasoning RL
```

而更像：

```text
                    MiMo Policy
                        │
      ┌─────────────────┼─────────────────┐
      ↓                 ↓                 ↓
   Coding            General          Tool / Agent
      ↓                 ↓                 ↓
 code harness      QA harness        agent harness
      │                 │                 │
      └──────── trajectories / rewards ───┘
                        ↓
                    one RL run
```

这让 RL 更像一个持续的大规模后训练阶段，而不是单一 benchmark 的专项强化。

---

# 16. MiMo-V2-Flash 对 V2.6 的重要背景

上一代 MiMo-V2-Flash 已经公开了大量和当前 V2.6 有关的基础设施设计：

### 模型

- 309B total parameters；
- 15B active parameters；
- 256K context；
- Hybrid Attention；
- Multi-Token Prediction（MTP）。

### RL / Agent 基础设施

- SGLang 作为 rollout / inference engine；
- Megatron-LM 作为 trainer；
- FP8 training / inference；
- Rollout Routing Replay（R3）；
- request-level prefix cache；
- fine-grained Data Scheduler；
- partial rollout；
- staleness-aware truncated importance sampling；
- Toolbox / Tool Manager。

这些内容说明 MiMo-V2.6 不是从零开始搭建异步 Agent RL，而是在已经非常成熟的 Agent RL infrastructure 上继续 scale。

---

# 17. 从 Cross Entropy 角度看 MiMo 的 RL Backward

最终无论系统多复杂，policy update 仍然回到：

```math
\log\pi_\theta(a_t|s_t)
```

假设一条 trajectory 的 advantage 为：

```math
A_i
```

最基础形式：

```math
L_i=-A_i\sum_t\log\pi_\theta(a_{i,t}|s_{i,t})
```

如果使用 PPO-style ratio：

```math
r_{i,t}=\frac{\pi_\theta(a_{i,t}|s_{i,t})}{\pi_{old}(a_{i,t}|s_{i,t})}
```

则：

```math
L_i=-\sum_t
\min\left(
 r_{i,t}A_i,
 \operatorname{clip}(r_{i,t},1-\epsilon,1+\epsilon)A_i
\right)
```

所以即使 MiMo 有：

```text
16 rollouts
+ dynamic sampler
+ async rollout
+ test-case reward
+ rubric reward
+ in-group credit assignment
```

最后还是在决定两件事：

1. **哪些 token/action 应该提高概率；**
2. **提高或降低多少。**

然后正常：

```python
loss.backward()
optimizer.step()
```

---

# 18. 当前最值得等待官方开源的细节

截至 2026-09-17，下面这些还没有完整公开：

1. MiMo-V2.6 的具体 policy optimization objective；
2. 是否直接沿用某种 GRPO / PPO 变体；
3. `Agentic In-group Credit Assignment` 的数学定义；
4. test-case reward 与 rubric reward 的融合方式；
5. Dynamic Sampler 的确切 acceptance rule；
6. 不同 harness 的 quota / sampling weight；
7. async rollout 与 trainer 的最大 staleness；
8. partial trajectory 如何跨 step 归因；
9. train/inference KL 的具体定义；
10. 是否继续使用 MiMo-V2-Flash 中的 R3、MTP rollout acceleration 等机制。

这些内容一旦公开，才能真正还原 MiMo-V2.6 的完整 RL recipe。

---

# 19. 最终心智模型

把 MiMo-V2.6 RL 压缩成一张图：

```text
                多个 Dataset / Harness
                         │
           quota / priority / history
                         │
                         ▼
                  Data Scheduler
                         │
                 candidate prompt
                         │
       ┌─────────────────┼─────────────────┐
       ↓                 ↓                 ↓
    rollout 1         rollout 2       ... rollout 16
       │                 │                 │
  long-horizon       long-horizon       long-horizon
     Agent              Agent              Agent
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
                       Grader
             test-case + rubric reward
                         │
                         ▼
                r1, r2, ..., r16
                         │
                         ▼
                 reward distribution
                         │
                         ▼
                  Dynamic Sampler
                    ↙          ↘
                reject        accept
                                │
                         凑够训练 groups
                                │
                                ▼
                    in-group credit assignment
                                │
                                ▼
                       advantage / weight
                                │
                                ▼
                  log π_new - log π_old
                                │
                                ▼
                    PPO/PG-style objective
                                │
                                ▼
                         loss.backward()
```

核心变化不是“RL 有一种完全不同的反向传播算法”。

真正的变化是：

> **从 ground-truth next token supervision，升级为大规模 exploration → environment feedback → reward → credit assignment → weighted log-prob gradient。**

---

## 参考资料

- MiMo RL Live Dashboard: https://mimo.xiaomi.com/rl/
- MiMo-V2-Flash Technical Report: https://arxiv.org/abs/2601.02780
- MiMo-V2-Flash GitHub: https://github.com/XiaomiMiMo/MiMo-V2-Flash
- IT之家对 MiMo-V2.6 RL 公开信息整理: https://www.ithome.com/1/003/555.htm
- 澎湃新闻对 MiMo-V2.6 RL 直播的报道: https://www.thepaper.cn/newsDetail_forward_34088356