<template>
  <div
    class="source-card"
    role="link"
    tabindex="0"
    :title="`打开《${source.book_title}》第 ${source.page} 页`"
    @click="openSource"
    @keydown.enter="openSource"
  >
    <div class="book-name">{{ source.book_title }}</div>
    <div class="chapter" v-if="source.chapter">{{ source.chapter }}</div>
    <div class="page">第 {{ source.page }} 页</div>
    <div class="excerpt">{{ source.excerpt }}</div>
    <div class="open-hint">打开原文 ↗</div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  source: {
    book_title: string
    chapter: string
    page: number
    excerpt: string
  }
}>()

function openSource() {
  const base = `/api/documents/${encodeURIComponent(props.source.book_title)}/file`
  // page > 0 时用 #page=N 让浏览器原生 PDF 阅读器定位到对应页
  const url = props.source.page > 0 ? `${base}#page=${props.source.page}` : base
  window.open(url, '_blank', 'noopener')
}
</script>

<style scoped>
.source-card {
  background: #fafafa;
  border: 1px solid #e8e8e8;
  border-radius: 10px;
  padding: 12px 14px;
  max-width: 320px;
  font-size: 12px;
  cursor: pointer;
  transition: border-color 0.15s, box-shadow 0.15s, background 0.15s;
  position: relative;
}

.source-card:hover,
.source-card:focus-visible {
  border-color: #4a90d9;
  background: #fff;
  box-shadow: 0 2px 8px rgba(74, 144, 217, 0.15);
  outline: none;
}

.book-name {
  font-weight: 700;
  color: #333;
  margin-bottom: 4px;
}

.chapter {
  color: #4a90d9;
  font-weight: 500;
  margin-bottom: 2px;
}

.page {
  color: #999;
  margin-bottom: 6px;
}

.excerpt {
  color: #666;
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 4;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

.open-hint {
  margin-top: 8px;
  color: #4a90d9;
  font-weight: 500;
  opacity: 0;
  transition: opacity 0.15s;
}

.source-card:hover .open-hint,
.source-card:focus-visible .open-hint {
  opacity: 1;
}
</style>
