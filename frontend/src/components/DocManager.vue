<template>
  <div class="doc-manager">
    <div class="upload-section">
      <h2>上传技术书籍</h2>
      <form @submit.prevent="upload" class="upload-form">
        <div class="field">
          <label>书名</label>
          <input v-model="bookTitle" placeholder="如：深入理解MySQL核心技术" />
        </div>
        <div class="field">
          <label>PDF 文件</label>
          <input type="file" accept=".pdf" @change="onFileChange" ref="fileInput" />
        </div>
        <button type="submit" :disabled="uploading || !bookTitle || !file">
          {{ uploading ? '上传中...' : '上传并入库' }}
        </button>
      </form>
      <div v-if="uploadMsg" :class="['upload-msg', uploadOk ? 'ok' : 'err']">
        {{ uploadMsg }}
      </div>
    </div>

    <div class="book-list">
      <h2>已入库书籍 ({{ books.length }})</h2>
      <div v-if="books.length === 0" class="empty">暂无书籍，请上传 PDF</div>
      <div v-for="book in books" :key="book.book_title" class="book-item">
        <div class="book-info">
          <div class="book-title">{{ book.book_title }}</div>
          <div class="book-meta">
            {{ book.page_count }} 页 · {{ book.chunk_count }} 个向量块
          </div>
        </div>
        <button class="delete-btn" @click="removeBook(book.book_title)">删除</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import axios from 'axios'

interface BookInfo {
  book_title: string
  file_path: string
  page_count: number
  chunk_count: number
}

const emit = defineEmits<{
  indexed: []
}>()

const bookTitle = ref('')
const file = ref<File | null>(null)
const fileInput = ref<HTMLInputElement>()
const uploading = ref(false)
const uploadMsg = ref('')
const uploadOk = ref(false)
const books = ref<BookInfo[]>([])

onMounted(() => {
  loadBooks()
})

async function loadBooks() {
  try {
    const { data } = await axios.get('/api/documents')
    books.value = data.books
  } catch (e) {
    console.error('加载书籍列表失败', e)
  }
}

function onFileChange(e: Event) {
  const target = e.target as HTMLInputElement
  const picked = target.files?.[0] || null
  file.value = picked
  // 自动用文件名（去扩展名）填充书名；若用户已填则不覆盖
  if (picked && !bookTitle.value.trim()) {
    bookTitle.value = picked.name.replace(/\.pdf$/i, '')
  }
}

async function upload() {
  if (!file.value || !bookTitle.value.trim()) return

  uploading.value = true
  uploadMsg.value = ''

  try {
    const form = new FormData()
    form.append('file', file.value)
    form.append('book_title', bookTitle.value.trim())

    const { data } = await axios.post('/api/documents/upload', form)
    uploadOk.value = data.status === 'indexed'
    uploadMsg.value = data.message

    if (uploadOk.value) {
      bookTitle.value = ''
      file.value = null
      if (fileInput.value) fileInput.value.value = ''
      loadBooks()
      emit('indexed')
    }
  } catch (e: any) {
    uploadOk.value = false
    uploadMsg.value = e.response?.data?.detail || e.message || '上传失败'
  } finally {
    uploading.value = false
  }
}

async function removeBook(title: string) {
  try {
    await axios.delete(`/api/documents/${encodeURIComponent(title)}`)
    loadBooks()
  } catch (e: any) {
    alert(e.response?.data?.detail || '删除失败')
  }
}
</script>

<style scoped>
.doc-manager {
  max-width: 700px;
  margin: 0 auto;
  padding: 30px 20px;
  height: 100%;
  overflow-y: auto;
}

h2 {
  font-size: 17px;
  font-weight: 600;
  margin-bottom: 16px;
  color: #333;
}

.upload-section {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 24px;
  margin-bottom: 24px;
}

.upload-form {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field label {
  font-size: 13px;
  color: #666;
  font-weight: 500;
}

.field input[type="text"],
.field input[type="file"] {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 6px;
  font-size: 14px;
}

button[type="submit"] {
  padding: 10px 20px;
  background: #4a90d9;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
  align-self: flex-start;
  transition: background 0.2s;
}

button[type="submit"]:hover:not(:disabled) {
  background: #3a7bc8;
}

button[type="submit"]:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.upload-msg {
  margin-top: 12px;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 13px;
}

.upload-msg.ok {
  background: #e8f5e9;
  color: #2e7d32;
}

.upload-msg.err {
  background: #fce4ec;
  color: #c62828;
}

.book-list {
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 12px;
  padding: 24px;
}

.empty {
  color: #999;
  text-align: center;
  padding: 20px;
}

.book-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0;
  border-bottom: 1px solid #f0f0f0;
}

.book-item:last-child {
  border-bottom: none;
}

.book-title {
  font-weight: 600;
  font-size: 14px;
}

.book-meta {
  font-size: 12px;
  color: #999;
  margin-top: 2px;
}

.delete-btn {
  padding: 4px 14px;
  background: transparent;
  color: #e57373;
  border: 1px solid #e57373;
  border-radius: 6px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s;
}

.delete-btn:hover {
  background: #e57373;
  color: #fff;
}
</style>
