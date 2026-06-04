<template>
  <div class="chat-window">
    <div class="chat-main">
      <div class="toolbar">
        <span class="session-tag" v-if="sessionId">会话: {{ sessionId.slice(0, 8) }}</span>
        <span class="session-tag" v-else>未选择会话（发送将自动创建）</span>
      </div>
      <div class="messages" ref="messagesEl">
        <div v-if="messages.length === 0" class="empty-hint">
          上传一本技术书籍 PDF 开始问答
        </div>
        <div
          v-for="(msg, i) in messages"
          :key="i"
          :class="['message', msg.role]"
        >
          <div v-if="msg.role === 'assistant' && msg.steps && msg.steps.length" class="agent-steps">
            <div v-for="(step, k) in msg.steps" :key="k" class="step">
              <span class="step-icon">{{ step.icon }}</span>
              <span class="step-text">{{ step.text }}</span>
            </div>
          </div>
          <div class="msg-content">{{ msg.content }}</div>
          <div v-if="msg.sources && msg.sources.length" class="sources-inline">
            <SourceCard
              v-for="(src, j) in msg.sources"
              :key="j"
              :source="src"
            />
          </div>
        </div>
      </div>
      <div class="input-area">
        <input
          v-model="input"
          @keydown.enter="send"
          placeholder="输入问题，如「InnoDB 的 MVCC 实现原理是什么？」"
          :disabled="loading"
        />
        <button @click="send" :disabled="loading || !input.trim()">发送</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, nextTick, watch, computed } from 'vue'
import SourceCard from './SourceCard.vue'
import { getMessages, type Source } from '../api/sessions'

interface AgentStep {
  icon: string
  text: string
}

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  steps?: AgentStep[]
}

const props = defineProps<{ sessionId: string | null }>()
const emit = defineEmits<{
  'message-sent': []
  'session-resolved': [id: string]
}>()

const messages = ref<Message[]>([])
const input = ref('')
const loading = ref(false)
const messagesEl = ref<HTMLElement>()

// 当外部切换 session 时，从后端加载历史
watch(
  () => props.sessionId,
  async (sid) => {
    messages.value = []
    if (!sid) return
    try {
      const history = await getMessages(sid)
      messages.value = history.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources,
        steps: [],
      }))
      scrollToBottom()
    } catch (e) {
      console.error('加载历史失败', e)
    }
  },
  { immediate: true },
)

const sessionId = computed(() => props.sessionId || '')

async function send() {
  const text = input.value.trim()
  if (!text || loading.value) return

  messages.value.push({ role: 'user', content: text })
  input.value = ''
  loading.value = true

  // 占位 assistant 消息：push 后必须从数组取响应式 proxy 引用，
  // 直接修改原始对象不会触发 Vue 响应式更新（这是 ref 的常见陷阱）
  messages.value.push({ role: 'assistant', content: '', steps: [], sources: [] })
  const assistantMsg = messages.value[messages.value.length - 1]
  // 流式状态：是否已进入"最终答案"阶段（最近一次事件是 tool_result，且之后没有再 tool_call）
  const streamState = { inAnswer: false, hasToolCall: false }
  scrollToBottom()

  try {
    const resp = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        session_id: props.sessionId,
        top_k: 3,
      }),
    })

    if (!resp.ok || !resp.body) {
      throw new Error(`HTTP ${resp.status}`)
    }

    const reader = resp.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    while (true) {
      const { value, done } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })

      // SSE 按 \n\n 分割
      const parts = buffer.split('\n\n')
      buffer = parts.pop() || ''
      for (const part of parts) {
        const line = part.trim()
        if (!line.startsWith('data:')) continue
        try {
          const payload = JSON.parse(line.slice(5).trim())
          handleEvent(payload, assistantMsg, streamState)
          scrollToBottom()
        } catch (err) {
          console.warn('parse SSE failed', err, line)
        }
      }
    }
  } catch (e: any) {
    assistantMsg.content = `错误: ${e.message || e}`
  } finally {
    loading.value = false
    scrollToBottom()
    emit('message-sent')
  }
}

function handleEvent(
  payload: any,
  msg: Message,
  state: { inAnswer: boolean; hasToolCall: boolean },
) {
  switch (payload.type) {
    case 'session': {
      // 后端确认/创建了 session_id，通知父组件同步
      if (payload.session_id) emit('session-resolved', payload.session_id)
      break
    }
    case 'tool_call': {
      const kw = payload.tool_kwargs ? JSON.stringify(payload.tool_kwargs) : ''
      msg.steps?.push({ icon: '🔧', text: `调用 ${payload.tool_name}(${kw})` })
      // 新一轮工具调用：中间 Thought 不再显示
      state.inAnswer = false
      state.hasToolCall = true
      msg.content = ''
      break
    }
    case 'tool_result': {
      msg.steps?.push({ icon: '📄', text: `工具返回（节选）：${payload.preview?.slice(0, 80) || ''}` })
      // 工具返回后接下来的 delta 视为最终答案阶段
      state.inAnswer = true
      msg.content = ''
      break
    }
    case 'delta': {
      // 无工具调用：所有 delta 都是答案；有工具调用：只在 tool_result 之后追加
      if (!state.hasToolCall || state.inAnswer) {
        msg.content += payload.data || ''
      }
      break
    }
    case 'sources': {
      msg.sources = payload.data
      break
    }
    case 'answer': {
      // 兜底：最终一次性覆盖（防止 delta 累积有问题）
      if (payload.data) msg.content = payload.data
      break
    }
    case 'error': {
      msg.content = `错误: ${payload.data}`
      break
    }
  }
}

function scrollToBottom() {
  nextTick(() => {
    if (messagesEl.value) {
      messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    }
  })
}
</script>

<style scoped>
.chat-window {
  height: 100%;
  display: flex;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  max-width: 900px;
  margin: 0 auto;
  width: 100%;
}

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 20px;
  background: #fafafa;
  border-bottom: 1px solid #eee;
  font-size: 12px;
  color: #888;
}

.session-tag {
  font-family: ui-monospace, Menlo, Consolas, monospace;
}

.reset-btn {
  margin-left: auto;
  padding: 4px 12px;
  background: transparent;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 12px;
  color: #666;
  cursor: pointer;
}

.reset-btn:hover:not(:disabled) {
  background: #f0f0f0;
}

.reset-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.agent-steps {
  background: #fafbfc;
  border: 1px solid #eef0f2;
  border-radius: 8px;
  padding: 8px 12px;
  margin-bottom: 8px;
  font-size: 12px;
}

.step {
  display: flex;
  gap: 6px;
  padding: 2px 0;
  color: #888;
  line-height: 1.5;
}

.step-icon {
  flex-shrink: 0;
}

.step-text {
  font-family: ui-monospace, Menlo, Consolas, monospace;
  word-break: break-all;
}

.messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
}

.empty-hint {
  text-align: center;
  color: #999;
  margin-top: 100px;
  font-size: 15px;
}

.message {
  margin-bottom: 20px;
  max-width: 85%;
}

.message.user {
  margin-left: auto;
}

.message.user .msg-content {
  background: #4a90d9;
  color: #fff;
  border-radius: 14px 14px 4px 14px;
  padding: 10px 16px;
}

.message.assistant .msg-content {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 14px 14px 14px 4px;
  padding: 10px 16px;
  line-height: 1.6;
  white-space: pre-wrap;
}

.sources-inline {
  margin-top: 8px;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.typing {
  color: #999;
  font-style: italic;
  padding: 10px 16px;
}

.input-area {
  padding: 16px 20px;
  background: #fff;
  border-top: 1px solid #e0e0e0;
  display: flex;
  gap: 10px;
}

.input-area input {
  flex: 1;
  padding: 10px 14px;
  border: 1px solid #ddd;
  border-radius: 10px;
  font-size: 14px;
  outline: none;
  transition: border-color 0.2s;
}

.input-area input:focus {
  border-color: #4a90d9;
}

.input-area button {
  padding: 10px 24px;
  background: #4a90d9;
  color: #fff;
  border: none;
  border-radius: 10px;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.2s;
}

.input-area button:hover:not(:disabled) {
  background: #3a7bc8;
}

.input-area button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
