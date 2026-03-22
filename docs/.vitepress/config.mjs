import { defineConfig } from 'vitepress'

// 若仓库是「项目页」而非 username.github.io，请把 base 改成 '/仓库名/'
// 例如仓库叫 my-blog：base: '/my-blog/'
export default defineConfig({
  title: '每日一记',
  description: '论文心得与日常记录',
  lang: 'zh-CN',
  base: '/vicliuuuu/',
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '论文心得', link: '/papers/' },
      { text: '日记', link: '/diary/' }
    ],
    sidebar: {
      '/papers/': [
        { text: '论文心得', link: '/papers/' }
      ],
      '/diary/': [
        { text: '日记索引', link: '/diary/' }
      ]
    },
    socialLinks: [],
    footer: {
      message: '个人记录',
      copyright: 'Copyright © 2025'
    },
    search: {
      provider: 'local'
    }
  }
})
