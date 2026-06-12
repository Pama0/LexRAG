"""会话管理路由：CRUD + 历史消息"""
from fastapi import APIRouter, HTTPException

from core.persistence import repositories as repo
from core.workflow.doc_query_service import DocQueryService
from core.persistence.db import get_session
from api.schemas import (
    CreateSessionRequest,
    MessageItem,
    MessageListResponse,
    RenameSessionRequest,
    SessionInfo,
    SessionListResponse,
    SourceRef,
)


def create_sessions_router(query_service: DocQueryService) -> APIRouter:
    router = APIRouter(prefix="/api/sessions", tags=["sessions"])

    @router.get("", response_model=SessionListResponse)
    async def list_sessions():
        async with get_session() as db:
            rows = await repo.list_sessions(db)
            sessions = []
            for r in rows:
                cnt = await repo.count_messages(db, r.id)
                sessions.append(SessionInfo(
                    id=r.id,
                    title=r.title,
                    created_at=r.created_at.isoformat(),
                    updated_at=r.updated_at.isoformat(),
                    message_count=cnt,
                ))
            return SessionListResponse(sessions=sessions)

    @router.post("", response_model=SessionInfo)
    async def create_session(payload: CreateSessionRequest):
        async with get_session() as db:
            row = await repo.create_session(db, title=payload.title or "新会话")
            return SessionInfo(
                id=row.id,
                title=row.title,
                created_at=row.created_at.isoformat(),
                updated_at=row.updated_at.isoformat(),
                message_count=0,
            )

    @router.get("/{session_id}/messages", response_model=MessageListResponse)
    async def get_messages(session_id: str):
        async with get_session() as db:
            sess = await repo.get_session(db, session_id)
            if sess is None:
                raise HTTPException(status_code=404, detail="会话不存在")
            rows = await repo.list_messages(db, session_id)
            items = [
                MessageItem(
                    id=m.id,
                    role=m.role,
                    content=m.content,
                    sources=[SourceRef(**s) for s in repo.parse_sources(m)],
                    created_at=m.created_at.isoformat(),
                )
                for m in rows
            ]
            return MessageListResponse(session_id=session_id, messages=items)

    @router.patch("/{session_id}", response_model=SessionInfo)
    async def rename(session_id: str, payload: RenameSessionRequest):
        async with get_session() as db:
            ok = await repo.rename_session(db, session_id, payload.title)
            if not ok:
                raise HTTPException(status_code=404, detail="会话不存在")
            sess = await repo.get_session(db, session_id)
            cnt = await repo.count_messages(db, session_id)
            return SessionInfo(
                id=sess.id,
                title=sess.title,
                created_at=sess.created_at.isoformat(),
                updated_at=sess.updated_at.isoformat(),
                message_count=cnt,
            )

    @router.delete("/{session_id}")
    async def remove(session_id: str):
        async with get_session() as db:
            ok = await repo.delete_session(db, session_id)
            if not ok:
                raise HTTPException(status_code=404, detail="会话不存在")
        # 一并清掉内存里的并发锁
        query_service.reset(session_id)
        return {"deleted": True, "session_id": session_id}

    return router
