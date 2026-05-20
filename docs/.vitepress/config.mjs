import { defineConfig } from 'vitepress'

// 若仓库是「项目页」而非 username.github.io，请把 base 改成 '/仓库名/'
// 例如仓库叫 my-blog：base: '/my-blog/'

// 这段 script 会在 <head> 里同步执行（早于页面正文渲染），
// 用于在「diary 路径 + 未解锁 / 已过期」时立刻给 <html> 加 class，
// 配合 theme/style.css 把内容藏起来，防止 SSR 出来的明文闪一下。
//
// 注意：下面的 hashes 数组、STORAGE_KEY 派生规则、TTL 必须和
// theme/diary-config.js 完全一致，否则 head 早期判断会和 Vue 端不一致。
const diaryGuardScript = `(function(){try{var hashes=['f8c56ebc4d6c4566bee5ad2725e26eff28ae7850e019d785a0bf53d62250fd52','fabfbb64e732c066b677152e6d39521a4b36912775f06142cef53bf9ca7eb689'];var v=hashes.map(function(h){return h.slice(0,8);}).join('-');var k='diary-unlock::'+v;var TTL=3600000;var raw=localStorage.getItem(k);var ts=raw?parseInt(raw,10):0;var ok=ts>0&&(Date.now()-ts)<TTL;if(location.pathname.indexOf('/diary/')!==-1&&!ok){document.documentElement.classList.add('diary-locked');}}catch(e){}})();`

export default defineConfig({
  title: '每日一记',
  description: '论文心得、日常记录与实验日志',
  lang: 'zh-CN',
  base: '/vicliuuuu/',
  head: [
    ['script', {}, diaryGuardScript]
  ],
  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '论文心得', link: '/papers/' },
      { text: '日记', link: '/diary/' },
      { text: '实验记录', link: '/experiments/' }
    ],
    sidebar: {
      '/papers/': [
        { text: '论文心得', link: '/papers/' }
      ],
      '/diary/': [
        { text: '日记索引', link: '/diary/' }
      ],
      '/experiments/': [
        { text: '实验日志', link: '/experiments/' }
      ]
    },
    socialLinks: [],
    footer: {
      message: '个人记录',
      copyright: 'Copyright © 2026'
    },
    search: {
      provider: 'local'
    }
  }
})
