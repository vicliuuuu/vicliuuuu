<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue'
import { useRoute } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import DiaryGate from './DiaryGate.vue'
import {
  lockedPathSegment,
  STORAGE_KEY,
  UNLOCK_TTL_MS,
  isUnlockValid
} from './diary-config.js'

const route = useRoute()

// 客户端 hydrate 完成前 isClient=false，避免 SSR 渲染密码门导致 hydration mismatch
const isClient = ref(false)
const unlocked = ref(false)
// 用一个 tick 让所有 reactive 计算都能感知「过期检查」的时间推进
const tick = ref(0)

let expireTimer = null

function readUnlockedFromStorage() {
  try {
    return isUnlockValid(localStorage.getItem(STORAGE_KEY))
  } catch (e) {
    return false
  }
}

// 显式依赖 route.path 让 computed 在 SPA 路由切换时重算；
// route.path 不带 base（比如 "/diary/2026-05-15.html"），用子串匹配最稳。
const isDiaryPath = computed(() => {
  return (route.path || '').indexOf(lockedPathSegment) !== -1
})

const showGate = computed(() => {
  // 触发 tick 依赖，让定时检查能让 computed 重算
  // eslint-disable-next-line no-unused-expressions
  tick.value
  if (!isClient.value) {
    return false
  }
  return isDiaryPath.value && !unlocked.value
})

function applyLockClass() {
  if (typeof document === 'undefined') {
    return
  }
  const shouldLock = isDiaryPath.value && !unlocked.value
  document.documentElement.classList.toggle('diary-locked', shouldLock)
}

function refresh() {
  unlocked.value = readUnlockedFromStorage()
  applyLockClass()
}

function onUnlock() {
  unlocked.value = true
  applyLockClass()
  scheduleExpireCheck()
}

// 周期性检查解锁是否过期：用户长时间停在日记页面，到点也会自动重新上锁
function scheduleExpireCheck() {
  if (expireTimer) {
    clearInterval(expireTimer)
    expireTimer = null
  }
  // 每分钟检查一次，比 TTL 小很多即可，1h TTL 用 60s 间隔最多多看 1 分钟
  expireTimer = setInterval(() => {
    const stillValid = readUnlockedFromStorage()
    if (unlocked.value && !stillValid) {
      unlocked.value = false
      applyLockClass()
    }
    tick.value++
  }, 60 * 1000)
}

onMounted(() => {
  refresh()
  isClient.value = true
  scheduleExpireCheck()
})

onBeforeUnmount(() => {
  if (expireTimer) {
    clearInterval(expireTimer)
    expireTimer = null
  }
})

// 路由切换（点击导航等 SPA 跳转）时重新检查锁状态并更新 class
watch(
  () => route.path,
  () => {
    if (!isClient.value) {
      return
    }
    refresh()
  }
)
</script>

<template>
  <DefaultTheme.Layout />
  <Teleport v-if="showGate" to="body">
    <DiaryGate @unlock="onUnlock" />
  </Teleport>
</template>
