<template>
  <aside class="session-list">
    <div class="header">
      <h2>会话</h2>
      <button class="new-btn" @click="createNew" title="新建会话">+ 新建</button>
    </div>
    <div class="list">
      <div v-if="sessions.length === 0" class="empty">暂无会话</div>
      <div
        v-for="s in sessions"
        :key="s.id"
        :class="['item', { active: s.id === currentId }]"
        @click="select(s.id)"
      >
        <div class="item-main">
          <template v-if="editingId === s.id">
            <input
              v-model="editingTitle"
              @keydown.enter="commitRename(s)"
              @keydown.esc="cancelRename"
              @blur="commitRename(s)"
              @click.stop
              ref="editingInput"
              class="rename-input"
            />
          </template>
          <template v-else>
            <div class="title" :title="s.title">{{ s.title }}</div>
            <div class="meta">{{ formatDate(s.updated_at) }} · {{ s.message_count }} 条</div>
          </template>
        </div>
        <div class="actions" @click.stop>
          <button class="icon-btn" @click="startRename(s)" title="重命名">✎</button>
          <button class="icon-btn danger" @click="remove(s)" title="删除">×</button>
        </div>
      </div>
    </div>
  </aside>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick } from 'vue'
import {
  listSessions,
  createSession,
  renameSession,
  deleteSession,
  type SessionInfo,
} from '../api/sessions'

const props = defineProps<{ currentId: string | null }>()
const emit = defineEmits<{
  select: [id: string]
  created: [id: string]
  deleted: [id: string]
}>()

const sessions = ref<SessionInfo[]>([])
const editingId = ref<string | null>(null)
const editingTitle = ref('')
const editingInput = ref<HTMLInputElement[] | HTMLInputElement>()

async function refresh() {
  try {
    sessions.value = await listSessions()
  } catch (e) {
    console.error('加载会话列表失败', e)
  }
}

onMounted(refresh)

defineExpose({ refresh })

function select(id: string) {
  emit('select', id)
}

async function createNew() {
  try {
    const sess = await createSession()
    await refresh()
    emit('created', sess.id)
  } catch (e: any) {
    alert('创建失败: ' + (e.message || e))
  }
}

function startRename(s: SessionInfo) {
  editingId.value = s.id
  editingTitle.value = s.title
  nextTick(() => {
    const el = Array.isArray(editingInput.value) ? editingInput.value[0] : editingInput.value
    el?.focus()
    el?.select()
  })
}

function cancelRename() {
  editingId.value = null
  editingTitle.value = ''
}

async function commitRename(s: SessionInfo) {
  if (editingId.value !== s.id) return
  const newTitle = editingTitle.value.trim()
  editingId.value = null
  if (!newTitle || newTitle === s.title) return
  try {
    await renameSession(s.id, newTitle)
    await refresh()
  } catch (e: any) {
    alert('重命名失败: ' + (e.message || e))
  }
}

async function remove(s: SessionInfo) {
  if (!confirm(`确认删除会话「${s.title}」？此操作不可撤销。`)) return
  try {
    await deleteSession(s.id)
    await refresh()
    emit('deleted', s.id)
  } catch (e: any) {
    alert('删除失败: ' + (e.message || e))
  }
}

function formatDate(iso: string): string {
  const d = new Date(iso)
  const now = new Date()
  const sameDay = d.toDateString() === now.toDateString()
  if (sameDay) {
    return d.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  }
  return d.toLocaleDateString('zh-CN', { month: '2-digit', day: '2-digit' })
}
</script>

<style scoped>
.session-list {
  width: 260px;
  background: #f7f7f8;
  border-right: 1px solid #e0e0e0;
  display: flex;
  flex-direction: column;
  height: 100%;
}

.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  border-bottom: 1px solid #e0e0e0;
}

.header h2 {
  font-size: 14px;
  font-weight: 600;
  color: #333;
}

.new-btn {
  padding: 4px 12px;
  background: #4a90d9;
  color: #fff;
  border: none;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
}

.new-btn:hover {
  background: #3a7bc8;
}

.list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.empty {
  color: #aaa;
  text-align: center;
  padding: 20px;
  font-size: 12px;
}

.item {
  display: flex;
  align-items: center;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 4px;
  transition: background 0.15s;
}

.item:hover {
  background: #eaeaeb;
}

.item.active {
  background: #d6e8f7;
}

.item-main {
  flex: 1;
  min-width: 0;
}

.title {
  font-size: 13px;
  color: #222;
  font-weight: 500;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.meta {
  font-size: 11px;
  color: #888;
  margin-top: 2px;
}

.rename-input {
  width: 100%;
  padding: 2px 4px;
  border: 1px solid #4a90d9;
  border-radius: 4px;
  font-size: 13px;
  outline: none;
}

.actions {
  display: none;
  gap: 4px;
  margin-left: 4px;
}

.item:hover .actions {
  display: flex;
}

.icon-btn {
  width: 22px;
  height: 22px;
  background: transparent;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  color: #666;
  font-size: 14px;
  line-height: 1;
}

.icon-btn:hover {
  background: #d0d0d0;
}

.icon-btn.danger:hover {
  background: #fce4ec;
  color: #c62828;
}
</style>
