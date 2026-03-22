# 个人站点（VitePress）

## 本地开发

```bash
cd website
npm install
npm run dev
```

浏览器打开终端里提示的地址（一般为 `http://localhost:5173`）。

## 上传到 GitHub

1. 在 GitHub 新建一个仓库（可设为 Public，别人才能直接访问 Pages）。
2. 在本机 `每日一记` 目录初始化并推送（若你希望**只发布网站**，可只把 `website` 文件夹当作仓库根目录）：

**方式 A：整个「每日一记」文件夹作为一个仓库**

```bash
cd 每日一记
git init
git add website
git commit -m "Add VitePress site"
git remote add origin https://github.com/你的用户名/仓库名.git
git branch -M main
git push -u origin main
```

**方式 B：`website` 单独成仓库（推荐，仓库更干净）**

把 `website` 里的内容作为仓库根目录（把其中文件复制到新文件夹或在该文件夹内 `git init`）。

## 让别人能访问（GitHub Pages）

1. 打开仓库 **Settings → Pages**。
2. **Build and deployment** 里 Source 选 **GitHub Actions**（本仓库已包含 `.github/workflows/deploy.yml`）。
3. 首次推送 `main`（或 `master`）分支后，等待 Actions 跑绿，Pages 会给出网址，形如：  
   `https://你的用户名.github.io/仓库名/`（若仓库名为 `用户名.github.io`，则域名为 `https://用户名.github.io/`）。

## 若网址带仓库子路径（项目页）

当站点地址是 `https://xxx.github.io/repo-name/` 时，请编辑 `docs/.vitepress/config.mjs`，把 `base: '/'` 改成：

```js
base: '/repo-name/'
```

然后重新推送，等待部署完成。

## 内容放哪里

| 类型     | 路径 |
|----------|------|
| 论文心得 | `docs/papers/` |
| 日记     | `docs/diary/` |

新增日记后，在 `docs/diary/index.md` 里加一条链接，方便索引。

## 若 Git 仓库根目录就是 `website` 里的内容

此时要把工作流里的路径改掉：删除所有 `website/` 前缀（`working-directory` 改为 `.`，`path` 改为 `docs/.vitepress/dist`，`cache-dependency-path` 改为 `package-lock.json`）。
