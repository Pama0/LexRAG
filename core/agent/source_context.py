"""请求级 source 收集容器（领域层）

工具调用时把检索到的 source_nodes 通过 contextvar 写入；
Web handler 在调用 Agent 前 begin_collection()，调用后 get_sources()。
contextvar 默认 task-local，子任务自动继承。

SourceRef 作为领域值对象定义在此，api 层从这里导入用于响应 DTO。
"""
from contextvars import ContextVar
from typing import Optional

from pydantic import BaseModel


class SourceRef(BaseModel):
    book_title: str
    chapter: str
    page: int
    excerpt: str  # 引用片段


_current_sources: ContextVar[Optional[list]] = ContextVar(
    "current_sources", default=None
)


def begin_collection() -> None:
    """请求开始时调用，重置收集器"""
    _current_sources.set([])


def add_sources(refs: list) -> None:
    """工具内调用，追加 SourceRef 列表"""
    bucket = _current_sources.get()
    if bucket is None:
        return
    bucket.extend(refs)


def get_sources() -> list:
    """请求结束时取出收集到的 sources（去重保序）"""
    bucket = _current_sources.get()
    if not bucket:
        return []
    seen = set()
    unique = []
    for s in bucket:
        key = (s.book_title, s.chapter, s.page, s.excerpt[:50])
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)
    return unique


def node_to_source_ref(node) -> SourceRef:
    """LlamaIndex node -> SourceRef"""
    meta = node.metadata or {}
    return SourceRef(
        book_title=meta.get("book_title", "未知"),
        chapter=meta.get("chapter", ""),
        page=meta.get("page", meta.get("page_start", 0)) or 0,
        excerpt=(node.get_content() if hasattr(node, "get_content") else node.text)[:300],
    )
