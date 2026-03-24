---
title: CLIP 阅读汇报
date: 2026-03-24
---

# CLIP 阅读汇报

## 论文信息

- 标题：Learning Transferable Visual Models From Natural Language Supervision
- 作者 / 会议或期刊：ICML
- 链接：[https://arxiv.org/abs/2103.00020](https://arxiv.org/abs/2103.00020)

## 一句话概括

本文首次通过大规模图像-文本对比学习，实现了无需任务特定微调的通用零样本图像分类能力。更关键的启示：监督信号可以来自数据本身，而不必依赖昂贵的人工标签

## 方法要点


传统模型通过固定类别标签学习图像到预定义类别的映射，类别权重硬编码在模型中; **CLIP则通过对比学习对齐图像与自然语言的语义空间，推理时用文本描述动态生成类别权重，实现开放词汇的零样本分类**

<details>
<summary>背景提要：传统的分类任务</summary>

以一个极简的3层CNN网络猫狗分类举例，先设定一个最小可运行模型，定义一个简单的网络：


```text
Input (32×32×3)
     ↓
Conv1 (3×3, stride=1, padding=1, out_channels=8)
     ↓
ReLU
     ↓
MaxPool (2×2)
     ↓
Conv2 (3×3, padding=1, out_channels=16)
     ↓
ReLU
     ↓
MaxPool (2×2)
     ↓
Conv3 (3×3, padding=1, out_channels=32)
     ↓
ReLU
     ↓
Global Average Pooling
     ↓
FC (2类：cat / dog)
     ↓
Softmax
```
0. 假设输入一张 32×32 的狗图像（有耳朵、鼻子、毛发），初始张量：(C×H×W)=(3×32×32)

1. 第一层Conv1(从像素->边缘)：output=8, kernel_size=3, 输出：(8×32×32)

这一层：此时还完全没有“狗”的概念

channel 1：水平边缘

channel 2：垂直边缘

channel 3：颜色变化

...

2. MaxPool(第一次降采样)

2×2 pooling，尺寸变为(8×16×16)

意义：降低分辨率；保留最强响应（重要特征）

3. 第二层Conv2(局部结构)

输入：(8×16×16)

输出：(16×16×16)

这一层开始组合边缘：耳朵轮廓；毛发纹理；眼睛局部结构

4. 再Pool: (16x8x8)

5. 第三层Conv3(语义雏形)

输入：(16x8x8)

输出：(32x8x8)

这一层开始接近语义，尽管网络不知道“这是狗脸”，只是统计上学到这种模式 → dog标签常出现；某些channel可能变成：

channel 5：狗脸模式

channel 12：猫耳朵模式

channel 20：鼻子形状

6. Global Average Pooling

对每个channel：把(32x8x8)变成(32)；得到x = [x₁, x₂, ..., x₃₂]

直觉理解：

x₅：狗脸模式出现强度

x₁₂：猫耳朵模式强度

x₂₀：鼻子模式强度

7. 分类：

全连接层（各自的权重连接，例如猫一个权重，狗一个权重, Wcat​∈R32, Wdog​∈R32）

->计算得分(scat​=Wcat​⋅x, sdog​=Wdog​⋅x)

->softmax(P(dog) = 0.95, P(cat) = 0.05)

```text
x（这张狗图） = [0.1, 0.3, ..., 2.5, ..., 0.2]
W_dog 在“狗脸”“鼻子”维度上权重大，W_cat 在“猫耳朵”维度上权重大
结果：s_dog = 4.2，s_cat = 1.1
```

**CNN把一张图像压缩成一个语义向量，然后用这个向量去匹配“类别方向”，每个 channel ≈ 一个“检测器”，例如某个channel对“狗脸”（“猫耳朵”）敏感**

**真正学习到的是权重值：所有图像共享同一套权重，而训练的目标是让这套权重能把不同图像映射到“可分的特征空间”，不同图像 → 不同的 x，但是整个CNN共用一套参数，所有图像共用**
     
</details>

<details>
<summary>背景提要：NLP带来的技术革命</summary>

**模型通过预测任务，把语言的“统计规律”编码进向量空间，使得语义、语法、知识都转化为“向量关系 + 概率分布”**

0. 以GPT-3为例，语言模型可以理解为：
```text
文本 → token → embedding → Transformer → 概率分布 → 生成文本**
```

1. token作为文本的最小单元，也是模型处理的基本输入单元，把自然语言变成模型可以处理的离散单位。例如：

```text
"The cat is playing"
→ ["The", "cat", "is", "play", "ing"]
```

2. 然后Embedding语义向量空间相当于把token映射到向量空间中的一个点，其核心性质是：通过向量的接近来表示语义的相似，这里相似的计算常用余弦相似度：

```text
cat   → [0.9, 0.3, 0.1]
dog   → [0.85, 0.25, 0.15]
table → [-0.4, 0.7, 0.2]
```

3. 整体的流程都应用到了自监督学习，训练目标是给定前面的词，去预测下一个词，例如：

```text
"The cat is on the" → mat
```

之所以称为自监督，是因为标签来自数据本身，不需要人工标注：

```text
输入：The cat is on the
标签：mat（自动生成）
```

模型在其中实现的过程可以总结为：预测错误 → 计算 loss → 反向传播 → 更新参数

4. 学习**语义**的核心就是通过共现关系来判断，所谓共现就是两个词经常在相似的上下文中出现，例如：

```text
cat ↔ mat（经常一起出现）
cat ↔ car（很少一起出现）
```

随之，在训练过程中：cat 和 mat → 向量越来越接近，cat 和 car → 向量越来越远，从本质上来说，语义就是通过大规模统计的共现结构

举一个更贴切的例子，cat/dog/car，模型如何判断cat和dog比car更相似？真正原因是他们在上下文中的使用方式相似程度不同：

cat的上下文：

```text
The cat is sleeping
The cat eats fish
The cat runs
The cat is on the mat
```

dog的上下文：

```text
The dog is sleeping
The dog eats meat
The dog runs
The dog is on the grass
```
car的上下文：
```text
The car is moving
The car is parked
The car is fast
The car is on the road
```

模型会根据除了这个关键词之外的词语进行综合判断，在训练时，模型要预测："The cat is ___"，模型要预测：sleeping / running / eating；那么这里同样如果把cat换为dog，需要预测几乎一样的词，为了降低loss，模型必须要让embedding(cat) ≈ embedding(dog)；但对于car, 预测的就变为moving / parked / fast，完全不同，所以embedding(cat) ≠ embedding(car)。

从数学上进行解释，模型不断优化：P(next word∣context)。如果两个词语可以互换：

```text
"The cat is sleeping"    (yes)
"The dog is sleeping"    (yes)
"The car is sleeping"    (wrong)
```

那么他们的embedding必须接近，否则模型无法用同一套参数进行预测，如果不能互换，embedding必须远离。

5. 学习**语法**的核心本质也是高频的统计规律。例如：

```text
He → is（常见）
He → are（几乎不出现）
```

但是在训练后：

```text
embedding(he) ≈ embedding(is)
embedding(he) ≠ embedding(are)
```

本质上所谓语法规则还是一种高频统计模式

6. 概率的来源，例如模型会输出：P(mat) = 0.4, P(table) = 0.3, P(car) = 0.05; 这个来源于embedding + 上下文 → 计算得分 → softmax → 概率，可以简单理解为：概率即当前上下文下的“合理程度”

7. 最简单的问题的产生：“What is the capital of France?”，模型首先会通过tokenization（分词）和 embedding（向量化）将这句话转换为向量表示:

```text
“What” → [0.2, 0.5, -0.1]
“is” → [0.3, 0.4, 0.1]
“the” → [0.1, 0.2, 0.4]
“capital” → [0.9, 0.1, 0.2]
“of” → [0.1, 0.3, -0.4]
“France” → [0.85, -0.2, 0.6]
```

然后从embedding空间获取语义信息，模型会通过“capital” 和 “France” 的 embedding 来推测它们之间的关联，最终将与 “capital of France” 高度相关的词（如 “Paris”）作为输出。例如：

```text
“capital” → [0.9, 0.1, 0.2]
“France” → [0.85, -0.2, 0.6]
“Paris” → [0.92, -0.1, 0.5]
```

模型会计算出 “Paris” 的嵌入向量与输入向量的相似度，→ cosine similarity: 高，表明“Paris”是一个合理的答案, 一方面，France ↔ Paris（强共现）；另一方面，capital ↔ city（语义约束）；最终选择 “Paris” 作为最符合上下文的答案。问答的本质还是条件语言生成。

进一步的，模型学会生成整个句子，生成式语言模型并不是直接给出整个句子的答案，而是逐步生成词语，知道生成完整的句子。这过程也是经历了模型计算概率的过程(softmax)


</details>

CLIP的动机源于对传统计算机视觉范式局限性的深刻反思——以往的视觉模型依赖于人工标注的封闭类别标签进行训练，不仅成本高昂，而且无法泛化到训练时未见过的新概念，严重制约了模型在开放世界中的适用性；与此同时，自然语言处理领域近年来通过在海量原始文本上进行自监督预训练（如BERT、GPT等）取得了革命性突破，证明了语言本身蕴含的强大语义学习能力，这启发CLIP作者提出一个关键洞见：如果人类能通过语言描述理解新物体，那么视觉系统是否也能以自然语言作为监督信号来学习视觉概念？由此，CLIP的核心思想便是摒弃固定类别标签，转而利用互联网上天然存在的大规模图像-文本对，通过构建一个双编码器架构（图像编码器与文本编码器），在共享的嵌入空间中以对比学习的方式对齐视觉与语言语义，使得模型在预训练阶段学会判断“哪段文字最能描述这张图”；这种对齐使得在下游任务中，无需任何微调，仅需将目标类别转化为自然语言提示（如“a photo of a dog”），用文本编码器生成对应的语义向量，再与图像特征计算相似度，即可实现开放词汇、零样本的图像识别，从而将视觉理解从封闭的标签空间解放到开放的人类语言空间。



## 一些想法



## 相关工作（可选）
[GRaD-Nav++: Vision-Language Model Enabled Visual Drone Navigation with Gaussian Radiance Fields and Differentiable Dynamics](https://arxiv.org/abs/2506.14009)
