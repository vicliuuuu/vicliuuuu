---
title: DINO 阅读汇报
date: 2026-04-09
---

# DINO 阅读汇报

## 论文信息

- 标题：Emerging Properties in Self-Supervised Vision Transformers
- 作者 / 会议或期刊：ICCV
- 链接：[https://arxiv.org/abs/2104.14294](https://arxiv.org/abs/2104.14294)

## 一句话概括

受到ViT的class token的注意力结果启发，提出的一种自监督学习框架，提出自监督学习与ViT架构的结合，不仅能产生强大的通用视觉表征，还能激发出 ViT 独有的、类似“对象意识”的能力

## 方法要点

### 思维启发来源

![image1](../images/DINO1.png)

上图很好的展示了 DINO 的灵感来源，这张图展示了在没有人工标注情况下，ViT的class token([CLS])的自注意力图本身就可以隐式的学习到物体的语义边界，从而实现高质量的、无需任何标注的物体分割，这是通过自监督方式训练的结果。虽然 [CLS] token 本身是一个无空间位置的全局向量，但在 Vision Transformer 的最后一层，它会通过自注意力机制计算自己对每个图像块（patch）token 的关注度——即生成一组标量注意力权重；由于这些 patch token 在输入序列中严格按图像空间顺序排列（如从左到右、从上到下），我们可以将 [CLS] 对应于各 patch 的注意力权重重新排列（reshape）成与原始图像网格一致的二维热力图；这张热力图直观地反映了 [CLS] 在做全局表征时“聚焦”于图像的哪些区域，而 DINO 论文发现，在自监督训练下，这种聚焦恰好高度集中在语义物体上，从而涌现出无需标注的物体分割能力。这种能力不是人物设计的，而是通过自然涌现的。

究其原因可以分为4个点：第一是自监督目标驱动语义一致性，DINO的目标是让不同增强试图下的特征一致，为了做到这一点，模型必须理解“同一物体”的概念，而忽略背景噪声的影响，这迫使模型学习对象级的不变表示；第二是由于ViT的全局建模能力，其具有全局交互能力，class token作为全局汇总节点，很容易能看到整个物体的完整结构；第三个是8x8的patch提供了丰富的空间细节，可以精准定位与分割；第四是因为无监督目标避免了类别的偏执，模型不会被“猫=毛茸茸+四条腿”的先验舒服，而是更纯粹的学习视觉显著性与结构一致性，能更具有泛化性。

对于监督学习，应该进行如下区分：

> 有监督学习使用人工标注的明确标签（如类别、边界框）作为监督信号；

> 自监督学习从数据本身构造监督信号（如预测被遮盖的部分、对比不同增强视图），无需任何人工标签；

> 弱监督学习则利用不完整、不精确或间接的标签（如图像级标签代替像素级标签、自然语言描述、标签噪声等）进行训练。

简单来说就是：有监督靠人标，自监督靠自己造任务，弱监督靠“粗糙”或“间接”的标签。

以ViT为模型架构举例：
| 方法/模型 | 模型架构 | 训练范式 | 是否需要人工标签？ | 监督信号来源 |
|----------|--------|--------|------------------|------------|
| 原始 ViT (2021) | ViT | 有监督 |  是 | ImageNet 类别标签 |
| DINO | ViT | 自监督 |  否 | 图像自身增强视图 |
| MAE | ViT | 自监督 |  否 | 被遮盖的图像块 |
| CLIP | ViT（图像编码器） | 弱监督/多模态 | （无需分类标签） | 配对的自然语言文本 |
| YOLOv8 | CNN 或 Hybrid | 有监督 |  是 | 边界框 + 类别标签 |

作者在abstract部分提出两大关键发现：**1.显式的语义分割信息：通过自监督学习训练的 ViT，其内部特征天然地包含了用于语义分割的明确信息。这种能力在有监督训练的 ViT 或任何卷积网络中都不明显或不存在。2.卓越的 k-NN 分类性能：这些自监督学到的特征质量极高，甚至可以直接用作k-近邻（k-NN），在 ImageNet 上达到了 78.3% 的 top-1 准确率（使用小型 ViT）。**

本文实现了一个self-**di**stillation with **no** labels(DINO)的自监督方法，并解释为一种无标签的自蒸馏方法，其中最关键的技术有三个：动量编码器、多裁剪训练策略以及在ViT中使用小尺寸图像块。

### 相关工作

<details>
<summary>1. Self-supervised learning 自监督学习</summary>

早期方法主要围绕实例判别，将每一张图片当作一个独立的类别，目标是让模型能区分任意两张不同的图，这里实现的方法是对同一张图做不同的数据增强，得到两个视图，要让模型认为它们相似，而对其他图认为不相似。这样做到的缺点是计算和开销巨大，无法扩展；进而有人提出了对比学习，其核心思想就是不再做分类，而是取比较特征维度之间的相似度，只要拉近正样本对，推开负样本对就行。

最新研究发现，其实根本不需要负样本，也不用区分不同图像，BYOL是里程碑式的工作，同一张图的两个增强视图分别经过两个网络，强制它们的输出特征尽可能一致就可以了，不用负样本，甚至后面发现两个网络完全对称时也可以工作，即只匹配正样本。

这里提到的BYOL，是Deepmind在2020年提出的自监督学习方法，它的工作流程很有趣，它通过两个神经网络，一个在线网络和一个目标网络。在线网络主要负责训练对象，通过梯度反向传播正常更新；而目标网络提供目标特征，不通过梯度更新，而是用在线网络的动量滑动平均进行更新，简单的说就是两者结构完全相同，参数更新的方式不同。

假设现在有一张原始图像x，步骤1：数据增强，对x做两次不同的增强，比如随机裁剪或色彩抖动得到两个视图：x₁ = Augment(x)，x₂ = Augment(x)；步骤2：双分支前向传播，把 x₁ 输入 online 网络 → 得到特征 z₁，把 x₂ 输入 target 网络 → 得到特征 z₂_target，当然也可以反过来（x₂ 进 online，x₁ 进 target），通常会做对称 loss；步骤3：在线网络网络后面接一个预测头（predictor，一个小 MLP）输出：q₁ = predictor(z₁)，目标网络网络后面只有投影头（projector，无预测头）输出：z₂_target；步骤4：计算损失，用MSE进行匹配，目标是让q₁ ≈ z₂_target（归一化后的向量）；步骤5：参数更新，在线网络网络（含预测头）：通过上述 loss 计算梯度，正常反向传播更新，目标网络网络不参与梯度计算！它的参数按如下方式更新：θtarget​←0.99⋅θtarget​+0.01⋅θonline

目标网络的滑动平均更新是相对较慢变化的过程，可以消除随机性，同时可以提供一个稳定的表示；预测头的存在防止了系统的表征坍塌，他要去拟合一个动态变化的目标

在前人基础上，主要是受到BYOL的启发，作者用了不同的相似性匹配损失函数，同时设定学生和教师网络结构完全相同。

</details>


<details>
<summary>2. Distilling the Knowledge in a Neural Network 自训练与知识蒸馏</summary>

在传统的 self-training 框架中，模型首先利用少量标注数据进行初始训练，然后将该模型应用于大量未标注数据，通过预测生成伪标签（pseudo-labels），再利用这些伪标签进行进一步训练，从而实现性能提升。这一过程的核心在于“标签传播”，即将有限的监督信息扩展到更大的数据分布上。根据伪标签形式的不同，这种传播可以分为两类：一类是硬标签（hard assignment），即为每个样本分配一个确定类别；另一类是软标签（soft assignment），即为每个样本分配一个概率分布。

当采用软标签时，这一过程在形式上与知识蒸馏高度一致。蒸馏的经典定义（如 Distilling the Knowledge in a Neural Network 所提出）是利用一个教师模型生成的概率分布作为监督信号，训练一个学生模型去逼近该分布。因此，可以将 soft pseudo-label 看作教师模型在未标注数据上的输出，而学生模型通过最小化分布差异（如 KL 散度）来学习这些知识。Xie 等人的工作进一步指出，这种机制本质上是在 self-training 框架中引入蒸馏，从而建立了两者之间的内在联系：self-training 是标签传播的视角，而蒸馏是分布匹配的视角，它们在软标签情形下是等价的。

在此基础上，DINO 将这一思想推进到一个更极端但也更有趣的情形：完全无标签学习。与传统 self-training 不同，这里不存在任何初始标注数据，因此“标签传播”的起点不再是人工标注，而是模型自身的预测。具体而言，DINO 不再依赖一个预训练的固定教师模型，而是在训练过程中动态构建教师网络。该教师通常由学生模型参数的指数滑动平均（EMA）得到，从而形成一个随训练逐步演化的“稳定目标”。在这种设定下，蒸馏不再是一个后处理步骤，而是被直接转化为训练目标本身，即通过对齐教师与学生在不同视图（augmentations）下的输出分布，实现表示学习。

此外，这种方法与 co-distillation 存在一定联系。co-distillation 通常指多个结构相同的模型之间相互蒸馏，即教师和学生是对等关系，彼此交换知识。而在 DINO 中，虽然教师与学生结构相同，但它们之间并非对称关系：教师参数由学生的历史参数通过平均得到，并不反向从学生中学习。这种“单向蒸馏 + 动态教师”的机制，使得教师在训练过程中充当一个缓慢变化的目标分布，从而提供稳定的优化信号。

从更本质的角度来看，这段话揭示了一个重要统一视角：无论是 self-training 还是 knowledge distillation，其核心都是通过构造一个“目标分布”来指导模型学习，而 DINO 的关键创新在于，这个目标分布可以完全由模型自身生成，而无需任何外部标签或预训练模型。这种机制使得蒸馏从“模型压缩工具”转变为“表示学习范式”，也正是其能够在自监督学习中取得显著效果的根本原因。

</details>

### 核心设计

结构上的共性：DINO 的整体框架（一个 online/student 网络 + 一个 target/teacher 网络）与当时主流的自监督方法（如 BYOL, MoCo v2）是一致的。这些方法都避免了负样本，通过两个网络之间的交互来学习。尽管结构相似，但作者强调，他们的核心思想和解读角度是知识蒸馏（Knowledge Distillation）。他们不是在做对比学习或动量编码器更新，而是在进行一种无标签的自蒸馏（self-distillation with no labels）。这是 DINO 的关键切入点。

作者提到知识蒸馏其实就是训练一个学生网络（student g_θs），使其输出 Ps 尽可能地匹配一个已给定的教师网络（teacher g_θt）的输出 Pt。输出形式：两个网络的原始输出（logits）都会通过一个 softmax 函数 转化为一个 K 维的概率分布（Ps 和 Pt）。K 通常是分类头的维度（例如 65536）。同时这里还有带有温度参数的softmax,温度 τ 的作用：高温度（τ 大）：会使 softmax 输出的分布更平滑、更均匀（uncertain），保留更多关于次优类别的信息，而低温度（τ 小）：会使 softmax 输出的分布更尖锐、更集中（confident），接近 one-hot 编码。DINO中通常设置 τt < τs。这意味着 teacher 的输出 Pt 更平滑，而 student 的输出 Ps 更尖锐。这样设计的好处是，平滑的 Pt 提供了更丰富的监督信号（包含了类别间的相对关系），而 student 需要努力学习去拟合这个复杂的分布，从而学到更鲁棒的特征。在标准的蒸馏中teacher是固定不变的，学习目标就是最小化Ps和Pt的交叉熵损失，其本质上衡量了两个概率分布之间的差异。

DINO的核心创新是适配到自监督上，DINO输入是同一张图像x的多个增强视图，通过设置全局视图以及局部视图，同时通过师生网络的不对称输入，Student：接收所有视图（2个全局 + N个局部）作为输入，Teacher：只接收2个全局视图作为输入，迫使 student 学习从局部细节（local views）中提取出能够与 teacher 的全局语义（global views）相匹配的表征。这极大地提升了模型学习层次化、鲁棒特征的能力，并且是 DINO 能产生优秀注意力图的关键。

最终，提出相关的损失函数，Pt(xg₁) 会与 Ps(xg₂) 以及所有 Ps(local_views) 计算损失，Pt(xg₂) 会与 Ps(xg₁) 以及所有 Ps(local_views) 计算损失，这确保了全局-全局（global-to-global）和全局-局部（global-to-local）的一致性都被优化。

在网络架构上，Student 和 Teacher 使用完全相同的网络架构，不过拥有不同的参数，同时，Student通过随机梯度下降正常更新，而Teacher不通过反向传播更新，它的参数是Student参数的指数移动平均(EMA，其实就是BYOL的动量更新策略)：θt ← m * θt + (1-m) * θs。这保证了 teacher 的输出非常稳定，为 student 提供了一个高质量、低噪声的学习目标。

此外，为避免出现没有负样本存在带来的模型坍缩出现，进行了关于Center机制的设计。作者发现将“居中”和“锐化”两个操作结合操作，它们就能达到一个完美的平衡。其实，居中可以防止单一维度主导，锐化可以防止均匀分布坍缩，这两个相反的力量相互制衡引导模型学习到一个既有多样性又有判别性的、有意义的输出分布。

Teacher是目标分布的提供者，要生成一个稳定、结构化、具有判别性的概率分布，作为学习目标，他不参与反向传播，t = t.detach()；通过EMA更新，慢变化，θt​←λθs​+(1−λ)θt​；输出经过centering + sharpening: t = softmax((t - C) / tpt), 去掉全局偏置后再增强判别性。

Student是表示学习的主体（Representation Learner），去拟合 teacher 的输出，从而学习输入数据的结构化表示，他参与反向传播，update(gs)；输出更平滑，高温softmax, s = softmax(s / tps); 跨视图匹配，学习视角不变性。

通过以上两个方式，解决了collapse坍缩的问题。


### 代码实现

```python 

# gs, gt: student and teacher networks
# C: center (K)
# tps, tpt: student and teacher temperatures
# l.m: network and center momentum rates

for x in loader: # load a minibatch x with n samples
    x1, x2 = augment(x), augment(x) #random views

    s1, s2 = gs(x1), gs(x2) # student output n-by-K
    t1, t2 = gt(x1), gt(x2) # teacher output n-by-K

    loss = H(t1, s2)/2 + H(t2, s1)/2
    loss.backward()

    update(gs)
    gt.params = l * gt.params + (1-l) * gs.params
    C = m * C + (1-m) * cat([t1, t2]).mean(dim=0)

def H(t, s):
    t = t.detach() # stop gradient
    s = softmax(s / tps, dim=1) 
    t = softmax((t - C) / tpt, dim=1) # center + sharpen
    return - (t * log(s)).sum(dim=1).mean()

```

1. 定义部分：

gs：student 网络（online network），参数可梯度更新。

gt：teacher 网络（target network），参数不可梯度更新，通过 EMA 更新。

C：一个长度为 K 的向量（K = 类别数 / token 数），称为 center（中心），用于稳定训练、防止输出分布坍缩到单一类别（如所有样本都预测为 class 0）。

tps, tpt：student 和 teacher 的 softmax 温度超参（通常 tpt > tps，让 teacher 输出更“平滑”，student 更“尖锐”以利于学习）。

l, m：动量系数（momentum rates）：l 控制 teacher 网络参数更新速度（典型值如 l=0.996）,m 控制 center 更新速度（典型值如 m=0.9）

2. 主训练循环部分：

每次从数据加载器取一个 batch（含 n 张图像）

对每张图做两次随机增强（如裁剪、颜色抖动、高斯模糊等），得到两个视图 x1, x2。

这是自监督学习的标准做法（对比学习/蒸馏都需要多视角一致性）。

3. 前向传播部分：

gs(x1)：student 网络对视图 x1 的输出（通常是 ViT 的 [CLS] token 或全局平均池化后的特征），维度为 n × K（K 是输出维度，例如 1000 或 65536）。

gt(x1)：teacher 网络对同一视图 x1 的输出，注意：teacher 不参与反向传播，仅提供目标信号。

同理得到 s2, t2（对应 x2）。

4. 损失计算：

这里 H(t, s) 是自定义的交叉熵函数（见下方定义），表示 teacher 输出作为 target，student 输出作为 prediction。H(t1, s2)：用 teacher 在 x1 上的输出 t1 作为目标，去拟合 student 在 x2 上的输出 s2，H(t2, s1)：反过来，用 t2 拟合 s1。除以 2 是为了对称平均 → 双向蒸馏（symmetric distillation）

防止单向拟合导致 student 过度模仿 teacher 的噪声；强制 student 学习跨视图的一致性（类似对比学习的思想）；提升鲁棒性与泛化能力

5. 反向传播

仅对 student 网络 gs 计算梯度并更新；

6. 参数更新

update(gs)：用 SGD（或 Adam）更新 student 网络参数。

gt.params = ...：teacher 参数 = 动量更新（EMA）。

C = ...：center 更新：将 t1 和 t2 拼接（cat([t1, t2])），形状变为 (2n) × K，沿 batch 维度取均值：(2n) × K → K，得到当前 batch 的平均输出向量，再用动量更新：C ← m·C + (1−m)·batch_mean，最终 C 是一个长期稳定的“输出中心”，用于防止模型坍缩。若没有 center，teacher 输出可能逐渐偏向某几个类别（比如所有样本都接近 class 0），导致 student 学到退化解。加入 center 后，损失函数中会减去这个平均值，迫使输出分布围绕 0 均衡展开（类似白化）。

7. H(t, s)核心损失函数

| 步骤 | 操作 | 目的 |
|------|------|------|
| `t = t.detach()` | 断开梯度 | 确保 teacher 输出不参与反向传播 |
| `s = softmax(s / tps, dim=1)` | student 输出软化 | 在此基础上除一个 `tps` 大 → softmax 先尖锐后平滑 |
| `t = softmax((t - C) / tpt, dim=1)` | teacher 输出软化 + 去中心化 | 在此基础上除一个 `tpt` 小 v→ 先平滑再尖锐；减去 `C` 防止坍缩 |
| `- (t * log(s)).sum(dim=1).mean()` | 交叉熵损失 | 标准 KL 散度形式：`H(t, s) = KL(t ‖ s)`，最小化 student 与 teacher 的分布差异 |

| 特性 | BYOL | DINO |
|------|------|------|
| Loss | MSE between projected features | KL divergence between softmax outputs |
| Predictor | 有（student 多一层 MLP） | 无（student 和 teacher 结构完全对称） |
| Teacher 更新 | EMA of student | EMA of student |
| Center | 无 | 有（关键防坍缩机制） |
| Temperature | 无 | 有（student & teacher 各自温度） |
| 输出空间 | 隐空间（feature space） | 概率空间（logit space） |
| 可视化效果 | — | Teacher attention 自然聚焦物体区域 |

DINO 在 logit 空间做蒸馏 + center + 温度调节，才使得 ViT 的注意力图展现出惊人的语义分割能力。DINO 通过让 student 网络在 softmax 概率空间中拟合一个动态构建的、带中心校正的 teacher 网络输出，实现了完全无标签的自蒸馏学习；其核心在于：对称交叉熵损失 + EMA teacher + moving center，三者共同防止坍缩、提升表征质量。

上述版本不包含multi-crop，在自监督学习中，crop（裁剪） 是数据增强的核心操作之一：将一张图随机裁剪出多个区域（如全局视图 + 局部视图），不同尺度/位置的裁剪带来不同粒度的语义信息，大裁剪（global view）：捕捉整体结构（如整只猫），小裁剪（local view）：聚焦局部细节（如猫耳朵、眼睛）。包含的版本中学生网络（student）接收 2 个 global crops + N 个 local crops（如 6 个），而老师网络（teacher）只接收 2 个 global crops。Teacher 只看全局视图 → 输出更稳定、语义更宏观（避免局部噪声干扰），Student 看全局 + 局部 → 被迫从局部细节中学习如何匹配老师的全局语义表征 → 强化特征的层次性与鲁棒性。

加上multi-crop之后可以清晰聚焦物体轮廓，接近监督模型

## 一些想法

从别人看不到的地方发掘宝藏，融合别人想不到的方法进行创新！

在DINO基础上，DINOv2使用了更大更好的数据集，同时结合了DINO损失和iBOT损失（随机进行掩码，学到更加深入的局部特征），并采用Sinkhorn-Knopp (SK) （Sinkhorn-Knopp算法是一种经典的迭代算法，其核心思想是通过交替对矩阵的行和列进行缩放（归一化），将一个非负矩阵转换为一个“双随机矩阵”）中心化方法来替代传统的移动平均中心化（在教师网络的MLP头（无论是DINO head还是iBOT head）输出原型分数后，先对其应用softmax。然后，在这个softmax结果上运行3次Sinkhorn-Knopp算法的迭代。这个过程会强制教师输出的概率分布在batch维度上变得更加均匀（uniform），防止模型坍缩到平凡解），同时加入KoLeo正则化防止坍缩（一个高质量的特征表示应该具有高熵，即特征点在嵌入空间中应尽可能分散，而不是聚集在一起形成簇，KoLeo正则化通过惩罚特征与其最近邻之间的距离来工作。这里选择特征：首先，从一个批次（batch）的数据中选取要正则化的特征。在DINOv2中，通常选择的是第一个全局裁剪（global crop）的[CLS] token。归一化：对这些选中的特征向量进行ℓ2归一化，将它们投影到单位超球面上。计算最近邻距离：对于批次中的每一个特征向量 xi，计算它到批次内所有其他特征向量的欧氏距离，并找到最近的邻居，记该距离为 dn,i。
dn,i = min (j≠i) ||xi - xj||，构建损失函数：KoLeo损失被定义为所有最近邻距离的对数的负平均值。L_koleo = - (1/n) * Σ log(dn,i)加入总损失：这个损失项会被加到模型的总训练损失中。在DINOv2中，其权重被设置为0.1，实现了最大化最小距离，促进均匀分布，同时防止坍缩）。

进而，2025年提出了DINOv3。在数据集上进一步进行了扩展的同时引入了Gram Anchoring方法，同时后处理策略增强了模型的灵活性。

## 4.16 补充

| 方面 | DINO | CLIP / LiT |
| :--- | :--- | :--- |
| 训练驱动力 | 自蒸馏 (KL散度) | 对比学习 (InfoNCE损失) |
| 负样本 | 无 | 有 (Batch内其他样本) |
| 余弦相似度的角色 | 训练后评估/推理时的“副产品”或“度量工具” | 训练过程中的“核心优化目标” |
| 如何学会区分 | 通过内部一致性学习，特征空间自然形成良好结构 | 通过显式地拉近正样本、推开负样本来塑造特征空间 |

DINO 是隐式学习，CLIP 是显式学习

## 相关工作

[DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193)

[DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](https://arxiv.org/abs/2412.16334)

[DINOv3](https://arxiv.org/abs/2508.10104)