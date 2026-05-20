// 生活日记账号 + 密码配置（支持多账号 + 解锁有时效）
//
// 校验规则：把用户输入的「用户名:密码」做 SHA-256，命中数组里任一一项即放行。
// 解锁后写入 localStorage 一条「时间戳」，UNLOCK_TTL_MS 之内不再要求登录，
// 超过有效期会自动重新弹出登录框。
//
// 增加 / 修改账号：
//   1) 算 hash：
//        node -e "console.log(require('crypto').createHash('sha256').update('账号:密码').digest('hex'))"
//   2) 把 hex 加进 credentialHashes 数组（追加 = 新增账号；替换 = 改账号）
//   3) 同步把这份 credentialHashes 数组复制到 docs/.vitepress/config.mjs 的
//      diaryGuardScript 里（必须保持完全一致，否则 head 早期脚本判断会和 Vue 端不一致）
//
// 注意：这是「前端伪密码」，仓库源码里的 markdown 是公开的。它只挡得住「点开网站
// 随便翻」的人，不要把它当成真正的加密。

// 账号列表（每个元素是 sha256("用户名:密码") 的小写 hex）
//   lyx / lyxlovelyj
//   lyj / lyjlovelyx
export const credentialHashes = [
  'f8c56ebc4d6c4566bee5ad2725e26eff28ae7850e019d785a0bf53d62250fd52',
  'fabfbb64e732c066b677152e6d39521a4b36912775f06142cef53bf9ca7eb689'
]

// 解锁有效期，超过这段时间需要重新登录
export const UNLOCK_TTL_MS = 60 * 60 * 1000 // 1 小时

// 路径前缀匹配，凡是 URL pathname 里包含这个子串都会被锁
export const lockedPathSegment = '/diary/'

// localStorage key 里嵌入一份从 hash 列表派生的版本串，
// 改账号 / 加账号都会让 STORAGE_KEY 改变，旧的解锁记录自动失效。
const VERSION = credentialHashes.map((h) => h.slice(0, 8)).join('-')
export const STORAGE_KEY = `diary-unlock::${VERSION}`

// 判断 localStorage 里取出的原始值是否仍在有效期内
export function isUnlockValid(rawValue) {
  if (!rawValue) {
    return false
  }
  const ts = parseInt(rawValue, 10)
  if (!Number.isFinite(ts) || ts <= 0) {
    return false
  }
  return Date.now() - ts < UNLOCK_TTL_MS
}
