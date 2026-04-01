---
title: ViT 阅读汇报
date: 2026-03-25
---

# ViT 阅读汇报

## 论文信息

- 标题：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- 作者 / 会议或期刊：ICLR
- 链接：[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

## 一句话概括

ViT具有里程碑式的作用，首次成功地将纯Transformer架构直接应用于图像识别任务。

## 方法要点

### 简介

Transformer凭借自注意力机制成为NLP的主流框架，它不再依赖RNN/CNN，而是通过全局建模词与词的关系，实现并行化和长距离依赖捕捉。尽管在2020年前，ResNet仍然是主流SOTA，但作者认为既然Transformer在NLP中“大力出奇迹”，那么如果不做复杂修改，只做最小适配，是否也能为CV领域带来新的曙光？作者提出了属于图像的Transformer技术：

关键技术就是图像分块(Patch Embedding): 将 HxWxC 的图像划分为 N=HW/(P^2) 个 PxP 的patch，

每个 patch 展开为 (P^2)C维向量，经过线性投影得到 embedding

图像经过上述处理，变为一个token序列，就像句子中的单词序列！“Image patches are treated the same way as tokens (words) in an NLP application.”

最开始的效果其实并不如ImageNet, 原因主要是Transformer 是完全数据驱动的，没有任何关于“空间局部性”或“平移不变性”的先验。在小数据上容易过拟合，泛化差。转折点随之来到：大数据改变一切！作者提出了革命性理论：当数据足够大时，模型可以从数据中自动学习到空间结构，无需人为设计归纳偏置！

ViT应运而生！ViT的三大贡献可以归纳维：1. 极简设计：直接将标准 Transformer 应用于图像 patch 序列，几乎不做修改；2. 颠覆认知：证明大规模数据可以替代 CNN 的归纳偏置，挑战了 CV 的基本假设；3. 开启新范式：为后续 MAE、Swin、DeiT 等工作铺路，推动 “纯 Transformer 视觉模型” 时代到来。

### 具体流程

![image1](../images/ViT1.png)

图中展示了ViT的经典结构，假设现在有一个224*224*3的图像，输入ViT处理。ViT的输入为：图像+标签

1. 图像分块，将224*224*3的图像进行分块处理，划分为patch_size=16*16的patch，所以对应的就会有14*14个patch。

2. 将patch进行展开，本来一个patch是16*16*3个向量，这里展开为一个768维向量。

3. 线性投影，将768维的向量投影到dim(例如256)特征维度上，这就等价于Transformer的embedding dim

4. 拼接class token，class token之后的作用是一个相当于集成了全局信息的token

5. 加入位置编码，class token设置为0，如这里一共197*256维位置编码，pos_embed.shape = (1, N+1, D)。这里和transformer有一点不一样，原来的transformer不可学习，有相对位置结构，是sin/cos的固定函数，但是这里是完全自由学习的位置编码，等同于ViT让模型自己学每个位置。

6. 送入transformer encoder

7. 取class token经过MLP做分类

**ViT没有decoder, 因为 ViT 不是序列生成任务，而是分类任务**；在图像生成/图像->文本时才会由decoder

### 代码实现

```python
import torch
import torch.nn as nn
import math


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PatchEmbedding(nn.Module):
    """Image to Patch Embedding"""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # 图像大小
        self.img_size = img_size
        self.patch_size = patch_size

        # patch的数量
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        # 步长为patch_size的卷积，不进行跨patch的信息融合；输出一个768维的14*14的张量
        # 这一步对应论文中的Linear Projection of Flattened Patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        # .flatten(2) 从第2维开始展平到最后，展开为(B,768,196)
        # .transpose(1,2) 调整维度顺序，每个块对应所在区域的768维信息
        x = self.proj(x).flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer block with Pre-LN"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 attn_drop=0., proj_drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       drop=proj_drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                          in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        # [class] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # Initialize weights
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, N, D)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, D)

        x = x + self.pos_embed  # (B, N+1, D)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # (B, D)
        logits = self.head(cls_out)  # (B, num_classes)
        return logits


'''
# ViT-Base/16 配置（最常用）
# 224*224的一个图像，每个patch为16*16，3通道输入，1000类，768个特征维度，Transformer Encoder的层数为12，特征头12个，每头64维，前馈网络的隐藏层相对于特征维数的4倍，在QKV中保留偏置项，注意力权重Dropout比例为0.1，防止注意力过于集中，提升泛化能力，Attention 输出和 MLP 输出上的 Dropout 比例为0.1，正则化，防止过拟合
model = VisionTransformer(
    img_size=224,
    patch_size=16,
    in_chans=3,
    num_classes=1000,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4.,
    qkv_bias=True,
    attn_drop=0.1,
    proj_drop=0.1
)

# 测试前向传播
x = torch.randn(2, 3, 224, 224)
logits = model(x)
print("Output shape:", logits.shape)  # torch.Size([2, 1000])
'''
```

## 一些想法

ViT的篇幅其实并不多，但是已经说明白了其核心思想，极具价值；它首次将Transformer应用到图像领域，它甚至没怎么改整体的框架，不过实现了前人没有做到的事情，这就是贡献！


## 相关工作

[Vision Transformer with Deformable Attention](https://arxiv.org/abs/2201.00520)
