---
title: LiT 阅读汇报
date: 2026-04-16
---

# LiT 阅读汇报

## 论文信息

- 标题：LiT: Zero-Shot Transfer with Locked-image text Tuning
- 作者 / 会议或期刊：CVPR
- 链接：[https://arxiv.org/abs/2111.07991](https://arxiv.org/abs/2111.07991)

## 一句话概括

LiT：Locked-image Tuning。不再重新从头开始训练图像编码器，而是教会文本编码器去理解一个已经很强大的图像编码器的特征向量。

## 方法要点

计算机视觉领域学习范式经历过三次关键跃迁：从有监督微调到零样本学习，从人工属性到自然语言监督，从端到端训练到高效解耦对齐。

奠基之作是迁移学习(Transfer Learning)，在2010年代初期至中期，其核心思想两阶段任务，首先是在一个大型通用数据集上预训练一个CNN模型，然后在特定的下游任务上进行微调。这种方法虽然非常成功，但是它无法处理训练时从未见过的新类别或新任务。

到了第二个阶段，实现突破的是零样本迁移(Zero-Shot Transfer)的兴起，这个阶段引入了自然语言作为桥梁，彻底改变了曾经的规则，代表作就是CLIP，其核心贡献就是利用Transformer网络框架，图像利用ViT进行embedding，文本进行Embedding处理后和图像在特征空间中匹配，利用对比学习同时训练一个图像编码器和一个文本编码器。实现了强大的零样本迁移能力，在面对新任务时仅通过自然语言提示输出，则可计算图像特征与所有类别文本特征的相似度就能完成分类。

第三阶段，就是LiT的时代，他提出了质疑与反思，像CLIP这种同时更新图像和文本编码器的方式是否最优策略？LiT发现，锁定一个强大的预训练好的图像编码器，只训练一个随机初始化的文本编码器，效果出奇的好。一方面，提供了更高的效率，避免了重新训练昂贵的视觉主干网络，节省了大量计算资源；另一方面，锁定的图像编码器保留了其在干净数据上学到的强大、鲁棒的视觉表征能力，不会被噪声较大的图文对数据污染，从而得以实现更好的泛化性，能实现更高的性能。

LiT不是提出一个全新的范式，而是对CLIP所开创的零样本迁移范式进行了一次精妙的优化。它揭示了解耦训练的重要性，并提供了一条更高效、更实用的路径来构建高性能零样本模型。

### 代码实现

```python

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    ViTModel,
    BertModel,
    BertConfig,
)
from datasets import load_dataset
from torchvision import transforms
import random

# ============================
# 1. 模型构建（LiT核心）
# ============================

image_model_name = "google/vit-base-patch16-224-in21k"
text_model_name = "bert-base-uncased"

# text encoder 从零开始（LiT核心）
text_config = BertConfig.from_pretrained(text_model_name)

base_model = VisionTextDualEncoderModel(
    vision_model=ViTModel.from_pretrained(image_model_name),
    text_model=BertModel(text_config),
    projection_dim=512,
)

# 冻结图像 encoder（LiT核心）
for p in base_model.vision_model.parameters():
    p.requires_grad = False


# ============================
# 2. LiT Wrapper（关键改进）
# ============================

class LiTModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # CLIP核心：可学习温度
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))

    def forward(self, batch):
        outputs = self.model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            return_dict=True,
        )

        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # 关键：归一化（cosine similarity）
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # temperature scaling
        logit_scale = self.logit_scale.exp()

        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text


# ============================
# 3. 数据处理（增强版）
# ============================

processor = VisionTextDualEncoderProcessor.from_pretrained(
    image_model_name,
    text_model_name
)

# 图像增强（关键）
image_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
])


def preprocess(example):
    image = example["image"].convert("RGB")
    image = image_transform(image)

    # 多caption随机采样（关键）
    caption = random.choice(example["captions"])["caption"]

    encoded = processor(
        images=image,
        text=caption,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    return {
        "pixel_values": encoded["pixel_values"][0],
        "input_ids": encoded["input_ids"][0],
        "attention_mask": encoded["attention_mask"][0],
    }


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
    }


# ============================
# 4. 数据集
# ============================

dataset = load_dataset(
    "ydshieh/coco_dataset_script",
    "2017",
    split="train[:2000]"   # 可以调大
)

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
dataset.set_format(type="torch")

dataloader = DataLoader(
    dataset,
    batch_size=64,   # 尽量大
    shuffle=True,
    collate_fn=collate_fn
)

# ============================
# 5. 损失函数
# ============================

def contrastive_loss(logits_per_image, logits_per_text):
    B = logits_per_image.size(0)
    labels = torch.arange(B, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2


# ============================
# 6. 训练
# ============================

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LiTModel(base_model).to(device)

# 关键：训练 text + projection + logit_scale
optimizer = torch.optim.AdamW(
    list(model.model.text_model.parameters()) +
    list(model.model.visual_projection.parameters()) +
    list(model.model.text_projection.parameters()) +
    [model.logit_scale],
    lr=5e-5,
    weight_decay=1e-4
)

model.train()

for epoch in range(5):
    total_loss = 0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}

        logits_i, logits_t = model(batch)
        loss = contrastive_loss(logits_i, logits_t)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")

print("Training finished!")

```

## 一些想法

这篇文章更注重于实验部分的内容，和RT-DETRv2很像，用数据说话。这个文章提出的冻结ViT的方法也为后面的工作奠基。

## 相关工作

[DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](https://arxiv.org/abs/2412.16334)