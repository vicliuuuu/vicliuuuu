<script setup>
import { ref, onMounted, nextTick } from 'vue'
import { credentialHashes, STORAGE_KEY } from './diary-config.js'

const emit = defineEmits(['unlock'])

const username = ref('')
const password = ref('')
const error = ref('')
const submitting = ref(false)
const userInputEl = ref(null)
const passInputEl = ref(null)

async function sha256Hex(text) {
  const buf = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text))
  return Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('')
}

async function onSubmit() {
  if (submitting.value) {
    return
  }
  if (!username.value || !password.value) {
    error.value = '请输入账号和密码'
    return
  }
  submitting.value = true
  error.value = ''
  try {
    const hash = await sha256Hex(`${username.value}:${password.value}`)
    if (credentialHashes.includes(hash)) {
      try {
        // 写入解锁时间戳，配合 isUnlockValid 实现 1h 有效期
        localStorage.setItem(STORAGE_KEY, String(Date.now()))
      } catch (e) {
        // localStorage 不可用时只在当前会话生效，下次仍要登录
        console.warn('[DiaryGate] localStorage write failed:', e)
      }
      emit('unlock')
    } else {
      error.value = '账号或密码不对，再想想'
      password.value = ''
      await nextTick()
      passInputEl.value?.focus()
    }
  } finally {
    submitting.value = false
  }
}

onMounted(() => {
  userInputEl.value?.focus()
})
</script>

<template>
  <div class="diary-gate">
    <div class="diary-gate__card">
      <div class="diary-gate__title">生活日记</div>
      <div class="diary-gate__subtitle">这一页是私密的，请登录后查看</div>
      <form class="diary-gate__form" @submit.prevent="onSubmit">
        <label class="diary-gate__field">
          <span class="diary-gate__label">账号</span>
          <input
            ref="userInputEl"
            v-model="username"
            class="diary-gate__input"
            type="text"
            autocomplete="username"
            placeholder="username"
            spellcheck="false"
            autocapitalize="off"
          />
        </label>
        <label class="diary-gate__field">
          <span class="diary-gate__label">密码</span>
          <input
            ref="passInputEl"
            v-model="password"
            class="diary-gate__input"
            type="password"
            autocomplete="current-password"
            placeholder="password"
            spellcheck="false"
          />
        </label>
        <button
          class="diary-gate__btn"
          type="submit"
          :disabled="submitting || !username || !password"
        >
          {{ submitting ? '校验中…' : '登录' }}
        </button>
      </form>
      <div v-if="error" class="diary-gate__error">{{ error }}</div>
      <div class="diary-gate__hint">
      </div>
    </div>
  </div>
</template>

<style scoped>
.diary-gate {
  position: fixed;
  inset: 0;
  z-index: 9999;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 24px;
  background: var(--vp-c-bg);
  background-image: radial-gradient(
      circle at 20% 20%,
      rgba(100, 108, 255, 0.08),
      transparent 40%
    ),
    radial-gradient(
      circle at 80% 80%,
      rgba(255, 100, 150, 0.08),
      transparent 40%
    );
}

.diary-gate__card {
  width: 100%;
  max-width: 380px;
  padding: 32px 28px;
  border-radius: 14px;
  background: var(--vp-c-bg-soft);
  border: 1px solid var(--vp-c-divider);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
}

.diary-gate__title {
  font-size: 22px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  margin-bottom: 6px;
}

.diary-gate__subtitle {
  font-size: 14px;
  color: var(--vp-c-text-2);
  margin-bottom: 20px;
}

.diary-gate__form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.diary-gate__field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.diary-gate__label {
  font-size: 12px;
  color: var(--vp-c-text-2);
  font-weight: 500;
}

.diary-gate__input {
  width: 100%;
  padding: 10px 12px;
  font-size: 14px;
  color: var(--vp-c-text-1);
  background: var(--vp-c-bg);
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  outline: none;
  transition: border-color 0.15s ease;
  box-sizing: border-box;
}

.diary-gate__input:focus {
  border-color: var(--vp-c-brand-1);
}

.diary-gate__btn {
  margin-top: 4px;
  padding: 10px 16px;
  font-size: 14px;
  font-weight: 500;
  color: #fff;
  background: var(--vp-c-brand-1);
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.15s ease, opacity 0.15s ease;
}

.diary-gate__btn:hover:not(:disabled) {
  background: var(--vp-c-brand-2);
}

.diary-gate__btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.diary-gate__error {
  margin-top: 12px;
  font-size: 13px;
  color: var(--vp-c-danger-1, #e11d48);
}

.diary-gate__hint {
  margin-top: 18px;
  font-size: 12px;
  color: var(--vp-c-text-3);
  line-height: 1.5;
}
</style>
