
from pydantic import BaseModel

from core.agent.source_context import SourceRef  # 领域值对象，re-export 供 DTO 使用


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # 会话 ID，区分独立的对话上下文
    book_titles: list[str] | None = None  # 用户手动选定的查询范围（硬约束，多选）；空则全库
    top_k: int = 3


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef] = []


class DocumentUploadResponse(BaseModel):
    filename: str
    book_title: str
    status: str  # "indexed" | "failed"
    chunk_count: int = 0
    message: str = ""


class DocumentInfo(BaseModel):
    book_title: str
    file_path: str
    page_count: int = 0
    chunk_count: int = 0
    indexed_at: str = ""


class DocumentListResponse(BaseModel):
    books: list[DocumentInfo] = []
    total_vectors: int = 0


# ==================== 会话历史 ====================

class SessionInfo(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[SessionInfo] = []


class CreateSessionRequest(BaseModel):
    title: str | None = None


class RenameSessionRequest(BaseModel):
    title: str


class MessageItem(BaseModel):
    id: int
    role: str
    content: str
    sources: list[SourceRef] = []
    created_at: str


class MessageListResponse(BaseModel):
    session_id: str
    messages: list[MessageItem] = []
