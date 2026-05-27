---
title: LoRA 阅读汇报
date: 2026-05-27
---

# LoRA 阅读汇报

## 论文信息

- 标题：LoRA: Low-Rank Adaptation of Large Language Models
- 作者 / 会议或期刊：ICLR
- 链接：[https://arxiv.org/abs/2106.09685](https://arxiv.org/abs/2106.09685)

## 一句话概括

Low-Rank Adaptation 低秩适应，也可以理解为用低秩约束完成的下游任务微调方法。Adaptation指的是把一个已经预训练好的大模型迁移到一个具体下游任务上，Low-Rank是指在这个方法上施加的约束，它强制规定，这次适应过程中权重的变化量 W 必须是一个低秩矩阵。

## 背景信息

### 低秩矩阵

秩衡量的是一个矩阵里独立信息的数量，比如现在有两个3x3的矩阵
```text
    1 2 3           1 2 3
A = 4 5 6       B = 2 4 6
    7 8 9           3 6 9
```
上述第一个矩阵的秩为2，因为第三行实际可以从前两行计算得到；第二个更是极端的低秩矩阵，这9个数实际只承载了一个方向的信息。所以一定程度上，秩=矩阵能被压缩成多窄，任何一个秩为r的矩阵M=R(d行k列)，都可以精准的被分解成两个矮矩阵的乘积：M(dxk)=B(dxr)·A(rxk)

只要r远小于d,k，那么就可以将原本的大矩阵分解为两个参数较小的矩阵，那么对应的存储和计算开销都直线下降，从而实现极致的压缩比。和字面意思一样，假设一个1000x1000的方阵，其理论最大的秩是1000，而如果它实际的秩是4，那就是典型的低秩。

整篇论文提出了一个很关键的假设：预训练大模型在适配某个下游任务时，权重的变化量具有很低的内在秩。

### 前置实验

> 随机子空间投影优化

以BERT为例，2018年Li提出的微调思路是将其全量110M的参数拉平成一个超长向量(1*110M)，预训练得到对应每一维的参数改变量，θ=θ0+Δθ，每一维都能够实现独立变化。

> 低自由度问题的发现

Aghajanyan在2020年提出，先随便挑d个固定的方向，然后规定，只允许Δθ在这d个方向上变化，其他方向都不变也就是上面提到的Δθ=P·z：直观理解就是不会全量调整所有的参数，而是选择其中的d歌维度，训练一个很小的向量z，然后用一个投影矩阵P将这个z向量映射到和模型参数一样大的维度，进而去更新模型参数，也就是说新参数=原始预训练参数+一个由z生成的小扰动，训练时梯度只更新z，不直接更新完整的模型权重。

```python
import torch

D = 110_000_000          # BERT 总参数量
d = 200                  # 选定的随机子空间维度

# 第一步:随机生成一个固定的投影矩阵,永远不更新
P = torch.randn(D, d, requires_grad=False)  # ← 冻住
P = P / P.norm(dim=0)                       # 列归一化

# 第二步:唯一可训练的变量,只有 200 个数
z = torch.zeros(d, requires_grad=True)

# 第三步:微调循环
optimizer = torch.optim.Adam([z], lr=1e-3)  # ← 注意只优化 z

for batch in dataloader:
    delta_theta = P @ z          # 把 200 维投影到 110M 维
    theta = theta_0 + delta_theta  # 套在预训练参数上
    
    loss = compute_loss(theta, batch)
    loss.backward()              # ← 梯度只会流到 z 上
    optimizer.step()
```

上述实验的结果令人震惊，只有在设置维度小于100时，性能才开始真正下降，也就是意味着在微调阶段，真正有用的参数也就是200个，剩下的参数完全是冗余的。

### 现存方法的问题

> Adapter类方法引入推理延迟

Adapter的思路就是在原模型的每一层之间插入小的瓶颈模块，就是在之前的模型层之间加入一个Adapter(降维-非线性-升维)以及残差模块，参数量通常不到之前的1%。这样的方法虽然的确降低了参数量，但是的确也引入了推理延迟，Adapter必须串行执行，无法利用硬件并行，增添的太多串行节点带来了太多的延迟。

> Prefix/Prompt Tuning难以优化+占据序列长度

这种方法的思路是不动模型权重，而是在输入前面拼上若干个虚拟可训练的token，这些token是直接学出来的连续向量，用它们去引导模型行为：类似于输入:  [P₁ P₂ ... Pₗ]  [x₁ x₂ ... xₙ]，P就是可训练的软提示，而X就是真实输入，这里只训练这些prefix向量。但是这里仍然有两个真实痛点：其一，prefix tuning不是参数越多能力越强，其性能可能会随着token的上升而下降；其二，这里额外引入的token会占用序列长度，这个是更根本的问题，一个transformer的最大序列长度是有限的，这里会影响原本的序列长度，会影响上下文的功能与结构。

> BitFit 只调Bias

这种方法的效率极高，但是性能明显更低。

## 介绍

LoRA 的解决方案：冻结原始权重 W0，在每个被选中的权重矩阵旁边并联一条低秩旁路 ΔθW=B·A ，前向时把两者相加：h = W0 x + B·A x。训练只更新 $B, A$；部署时可以把 BA 合并进 W0，恢复成单一矩阵，实现零额外延迟。

LoRA并没有像之前的方法那样，将参数拉平成1xN的矩阵，而是保留了W的形状(768*768，分别对应输入的维度以及输出的维度)。这里对独立的W做矩阵的低秩分解，极大的降低了参数量，最后得到的是一堆A, B的矩阵对

```text
全量微调:                          LoRA:
   x                                  x
   │                                  ├─────────────┐
   ▼                                  ▼             ▼
[ W (全部可训练) ]               [ W₀ (冻结) ]   [ B·A (可训练) ]
   │                                  │             │
   ▼                                  └─────┬───────┘
   h                                        ▼
                                           h

Aghajanyan 视角:
                                 ┌──────────────────────────────┐
所有权重拉平成 1×N:    θ ∈ ℝ^N  =│xxxxxxxxxxxxxxxxxxxxxxxxxxxx │
                                 └──────────────────────────────┘
                                              │
                            约束在 200 维子空间│ Δθ = P·z
                                              ▼
                                      z ∈ ℝ²⁰⁰ (唯一可训)

--------------------------------------------------------------------------   

LoRA 视角:
                ┌─────────┐    ┌─────────┐    ┌─────────┐
保留每个 W 形状: │  Wq_1   │    │  Wv_1   │ …  │  Wv_12  │
                │ 768×768 │    │ 768×768 │    │ 768×768 │
                └─────────┘    └─────────┘    └─────────┘
                     │              │              │
              每个独立加 LoRA 约束:  ΔW = B·A, rank≤8
                     ▼              ▼              ▼
                  (A,B)对         (A,B)对        (A,B)对
                  各自可训         各自可训        各自可训     

--------------------------------------------------------------------------   

   d×k 大矩阵       =    d×r 瘦长条     ×    r×k 矮扁条
┌──────────────┐        ┌──┐              ┌──────────────┐
│              │        │  │              │              │
│     ΔW       │   =    │ B│      ×       │      A       │
│   (秩 ≤ r)   │        │  │              └──────────────┘
│              │        │  │              
└──────────────┘        └──┘                                             
```
1. 在整个过程中，W0全程冻结，训练过程中只是在前向通道中被使用；

2. 旁路是被加上去的，是作为低秩修正，实现了训练时的灵活训练以及推理时的零延迟；

3. 只训练A, B两个矩阵，这里BA相乘后对应的就是秩为r的矩阵，这个就是低秩矩阵的来源；

4. 训练结束后回到一个W，这个和原始模型架构一模一样；

5. 初始化A矩阵使用了高斯初始化，B矩阵等于0；防止对模型产生破坏；同时在使用时添加了一个缩放系数α，h = W₀ x + (α/r) · B A x，以降低调参成本；

6. 实验证明，将LoRA加在Wq+Wv上效果效果最好；这里比如需要微调的参数矩阵是768*768，那么这里对应的B以及A就应该分别是768*m以及m*768的矩阵，这里的m默认是8，按照经验，任务差距越大，所需的m也就越大

```python
# Step 1: 加载预训练模型
model = load_pretrained_model()
freeze_all_parameters(model)

# Step 2: 给目标层注入 LoRA 旁路
for layer in model.transformer.layers:
    for name in ['q_proj', 'v_proj']:   # 论文推荐
        old_linear = getattr(layer.attention, name)
        lora_linear = LoRALinear(
            old_linear.in_features, 
            old_linear.out_features, 
            r=4, alpha=4
        )
        lora_linear.weight.data = old_linear.weight.data   # 拷贝 W₀
        lora_linear.weight.requires_grad = False           # 冻结
        # A 高斯初始化, B 零初始化 (LoRALinear 内部已处理)
        setattr(layer.attention, name, lora_linear)

# Step 3: 训练 (优化器只看到 LoRA 参数)
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],   # 只剩 A, B
    lr=3e-4
)
for batch in dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()

# Step 4: 部署 - 合并权重
for m in model.modules():
    if isinstance(m, LoRALinear):
        m.merge()       # W ← W₀ + (α/r) · B A
# 现在 model 和一个普通 fine-tuned 模型完全等价

# Step 5: 推理
output = model(input)   # 零额外延迟
```

## 一些想法

LoRA 的核心思想用一行公式就能讲清，但在工程层面（无延迟、可热切换、参数压缩 10000×）几乎把同时代所有 PEFT 方法的痛点一次性解决了。它能成为后 GPT-3 时代事实上的微调标准，绝非偶然。

从我的认知出发，LoRA 的真正威力在于**它把"模型权重"和"任务行为"解耦了**。预训练模型从此可以被看成一个共享的"底座 OS"，每个下游任务只是一份很小的"任务补丁"。这种解耦在 LLM 服务化、Agent 多任务、个性化推荐等场景里都被反复证明是极重要的工程范式。

一个值得继续追问的点：LoRA 假设 $\Delta W$ 低秩，但有没有任务**真的不低秩**？论文自己也提到——比如下游任务用的是和预训练完全不同的语言时，可能就需要更大的 $r$。这给后续工作（AdaLoRA 自适应分配 rank、QLoRA 加量化、DoRA 分解方向与幅度）留了非常多的扩展空间。

跟 LiT 那篇文章一对比有意思：LiT 是"冻结视觉编码器、只训文本端"，LoRA 是"冻结整个模型、只训低秩旁路"——两者在思想上是同一类——**与其重训整个大模型，不如锁住它，再在外围加一点点小的、专门为任务而生的可训练组件**。这种"锁主体 + 加小补丁"的范式可能是这一代基础模型时代最重要的方法论。


## 相关工作
- [Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning](https://arxiv.org/abs/2012.13255)
- [AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2303.10512)
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)
- [DoRA: Weight-Decomposed Low-Rank Adaptation](https://arxiv.org/abs/2402.09353)