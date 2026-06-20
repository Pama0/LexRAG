<template>
  <div class="app">
    <header class="header">
      <h1>LibraryRAG</h1>
      <span class="subtitle">书籍知识库助手</span>
      <nav class="nav">
        <button :class="{ active: view === 'chat' }" @click="view = 'chat'">问答</button>
        <button :class="{ active: view === 'docs' }" @click="view = 'docs'">文档管理</button>
      </nav>
    </header>
    <main class="main">
      <template v-if="view === 'chat'">
        <SessionList
          ref="sessionListRef"
          :current-id="currentSessionId"
          @select="onSelectSession"
          @created="onCreatedSession"
          @deleted="onDeletedSession"
        />
        <ChatWindow
          :session-id="currentSessionId"
          @message-sent="refreshSessionList"
          @session-resolved="onSessionResolved"
        />
      </template>
      <DocManager v-else-if="view === 'docs'" @indexed="view = 'chat'" />
    </main>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import ChatWindow from './components/ChatWindow.vue'
import DocManager from './components/DocManager.vue'
import SessionList from './components/SessionList.vue'

const view = ref<'chat' | 'docs'>('chat')
const currentSessionId = ref<string | null>(null)
const sessionListRef = ref<InstanceType<typeof SessionList>>()

function onSelectSession(id: string) {
  currentSessionId.value = id
}

function onCreatedSession(id: string) {
  currentSessionId.value = id
}

function onDeletedSession(id: string) {
  if (currentSessionId.value === id) {
    currentSessionId.value = null
  }
}

function onSessionResolved(id: string) {
  // ChatWindow 在发首条消息时由后端自动创建了会话，把 id 回填
  if (currentSessionId.value !== id) {
    currentSessionId.value = id
  }
  refreshSessionList()
}

function refreshSessionList() {
  sessionListRef.value?.refresh()
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #f5f5f5;
  color: #333;
}

.app {
  height: 100vh;
  display: flex;
  flex-direction: column;
}

.header {
  background: #1a1a2e;
  color: #e0e0e0;
  padding: 12px 24px;
  display: flex;
  align-items: center;
  gap: 16px;
}

.header h1 {
  font-size: 20px;
  font-weight: 700;
}

.subtitle {
  color: #888;
  font-size: 13px;
}

.nav {
  margin-left: auto;
  display: flex;
  gap: 8px;
}

.nav button {
  background: transparent;
  color: #888;
  border: 1px solid #444;
  padding: 6px 16px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 13px;
  transition: all 0.2s;
}

.nav button:hover {
  border-color: #666;
  color: #ccc;
}

.nav button.active {
  background: #16213e;
  border-color: #4a90d9;
  color: #4a90d9;
}

.main {
  flex: 1;
  overflow: hidden;
  display: flex;
}
</style>
