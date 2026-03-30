# 论文心得

在这里用 Markdown 写读论文的笔记。每篇可以单独一个文件，便于版本管理与搜索。


## 神作

- [Transformer](./Transformer.md)

## 论文

- [Grad-nav](./grad-nav.md)
- [ViT](./ViT.md)
- [CLIP](./CLIP.md)
- [OWL-ViT](./OWL-ViT.md)
- [DETR](./DETR.md)
- [RTDETR](./RTDETR.md)


## 怎么新增一篇

在 `website/docs/papers/` 下新建 `某论文标题.md`， front matter 示例：

```yaml
---
title: 论文标题（可选）
date: 2025-03-22
---
```

写完后在本页或侧边栏里加上链接（可在 `.vitepress/config.mjs` 的 `sidebar` 里维护列表）。


