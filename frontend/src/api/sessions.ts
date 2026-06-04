import axios from 'axios'

export interface Source {
  book_title: string
  chapter: string
  page: number
  excerpt: string
}

export interface SessionInfo {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface MessageItem {
  id: number
  role: 'user' | 'assistant'
  content: string
  sources: Source[]
  created_at: string
}

export async function listSessions(): Promise<SessionInfo[]> {
  const { data } = await axios.get('/api/sessions')
  return data.sessions
}

export async function createSession(title?: string): Promise<SessionInfo> {
  const { data } = await axios.post('/api/sessions', { title: title ?? null })
  return data
}

export async function getMessages(sessionId: string): Promise<MessageItem[]> {
  const { data } = await axios.get(`/api/sessions/${sessionId}/messages`)
  return data.messages
}

export async function renameSession(sessionId: string, title: string): Promise<SessionInfo> {
  const { data } = await axios.patch(`/api/sessions/${sessionId}`, { title })
  return data
}

export async function deleteSession(sessionId: string): Promise<void> {
  await axios.delete(`/api/sessions/${sessionId}`)
}
