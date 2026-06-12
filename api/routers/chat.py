"""Chat 路由：所有问答走 Agent 模式，历史用 SQLite 持久化"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from core.persistence import repositories as repo
from core.workflow.doc_query_service import DocQueryService
from core.workflow.doc_workflow import (
    AnswerDeltaEvent,
    RetrievalDoneEvent,
    RetrievalStartEvent,
)
from core.persistence.db import get_session
from api.schemas import ChatRequest, ChatResponse, SourceRef
from core.agent.source_context import node_to_source_ref

logger = logging.getLogger(__name__)


async def _ensure_session(session_id: Optional[str]) -> str:
    """确保 session 存在：传入 None 或不存在的 id 都自动创建一个新会话，返回最终 id"""
    async with get_session() as db:
        if session_id:
            sess = await repo.get_session(db, session_id)
            if sess is not None:
                return sess.id
        # 新建
        new_sess = await repo.create_session(db)
        return new_sess.id


def create_chat_router(query_service: DocQueryService) -> APIRouter:
    """工厂函数：注入 query_service（DocQueryWorkflow 装配）依赖"""
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """非流式问答：workflow 跑完返回完整答案 + 来源"""
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        session_id = await _ensure_session(req.session_id)
        logger.info("chat: session=%s msg=%r", session_id, req.message[:50])

        lock = query_service.get_lock(session_id)
        async with lock:
            # 从 DB 加载历史构造 memory
            async with get_session() as db:
                history = await repo.list_messages(db, session_id)
            memory = query_service.build_memory(history)

            try:
                # scope 由 book_titles 直接传入 workflow（不再走 contextvar）
                result = await query_service.run_handler(
                    query=req.message,
                    memory=memory,
                    book_titles=req.book_titles,
                )
            except Exception as e:
                logger.exception("Workflow run failed")
                raise HTTPException(status_code=500, detail=f"查询执行失败: {e}")

            answer_text = str(getattr(result, "response", result))
            sources = _nodes_to_sources(getattr(result, "source_nodes", []) or [])

            # 写入 user + assistant 消息到 DB
            await _persist_pair(
                session_id=session_id,
                user_msg=req.message,
                assistant_msg=answer_text,
                sources=sources,
                is_first_in_session=(len(history) == 0),
            )

        return ChatResponse(answer=answer_text, sources=sources)

    @router.post("/chat/stream")
    async def chat_stream(req: ChatRequest):
        """流式问答：SSE 推送 Agent 事件 + 持久化"""
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        session_id = await _ensure_session(req.session_id)
        logger.info(
            "chat_stream: session=%s msg=%r",
            session_id, req.message[:50],
        )

        lock = query_service.get_lock(session_id)

        async def event_generator():
            # 第一条事件告诉前端最终 session_id（可能是后端新建的）
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            async with lock:
                async with get_session() as db:
                    history = await repo.list_messages(db, session_id)
                memory = query_service.build_memory(history)
                is_first = len(history) == 0

                try:
                    # scope 由 book_titles 直接传入 workflow（不再走 contextvar）
                    handler = query_service.run_handler(
                        query=req.message,
                        memory=memory,
                        book_titles=req.book_titles,
                    )

                    async for ev in handler.stream_events():
                        if ev.__class__.__name__ != "AnswerDeltaEvent":  # 跳过逐 token 增量，避免刷屏
                            logger.info("WORKFLOW EVENT %s\n  %s", ev.__class__.__name__, _debug_dump(ev))
                        payload = _format_event(ev)
                        if payload is not None:
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    final = await handler
                    final_text = str(getattr(final, "response", final))
                    logger.info("chat_stream done; final length=%d", len(final_text))

                    sources = _nodes_to_sources(getattr(final, "source_nodes", []) or [])
                    yield f"data: {json.dumps({'type': 'sources', 'data': [s.model_dump() for s in sources]}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'answer', 'data': final_text}, ensure_ascii=False)}\n\n"

                    await _persist_pair(
                        session_id=session_id,
                        user_msg=req.message,
                        assistant_msg=final_text,
                        sources=sources,
                        is_first_in_session=is_first,
                    )

                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                except Exception as e:
                    logger.exception("Workflow stream failed")
                    yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return router


def _debug_dump(ev) -> str:
    """把 workflow 事件的关键字段抽出来写日志，便于查看检索/合成轨迹。

    不取 delta（逐 token 增量，会刷屏；完整答案见 final_text）。
    """
    parts = []
    for attr in ("query", "count", "intent", "category", "rewritten_query"):
        val = getattr(ev, attr, None)
        if val:
            parts.append(f"{attr}={val!s}")
    return "\n  ".join(parts)


def _nodes_to_sources(nodes: list) -> list[SourceRef]:
    """workflow 结果的 source_nodes → SourceRef 列表（去重保序）。

    取代原 source_context.get_sources：检索改在 workflow 内直接做，
    source_nodes 随结果带出，不再走 contextvar。
    """
    seen = set()
    unique: list[SourceRef] = []
    for n in nodes:
        ref = node_to_source_ref(n)
        key = (ref.book_title, ref.chapter, ref.page, ref.excerpt[:50])
        if key in seen:
            continue
        seen.add(key)
        unique.append(ref)
    return unique

async def _persist_pair(
    session_id: str,
    user_msg: str,
    assistant_msg: str,
    sources: list[SourceRef],
    is_first_in_session: bool,
) -> None:
    """写入一对消息到 DB；首条 user 消息会触发自动标题"""
    async with get_session() as db:
        await repo.add_message(
            db, session_id, role="user", content=user_msg,
            auto_title_from_first=is_first_in_session,
        )
        await repo.add_message(
            db, session_id, role="assistant", content=assistant_msg, sources=sources,
        )


def _format_event(ev) -> dict | None:
    """把 DocQueryWorkflow 流式事件映射成前端已有的 SSE payload。

    复用前端的 tool_call→tool_result→delta 状态机（前端零改动）：
    - RetrievalStartEvent → tool_call（前端显示"调用检索"）
    - RetrievalDoneEvent  → tool_result（前端据此进入"答案阶段"）
    - AnswerDeltaEvent    → delta（逐 token 流入答案区）
    """
    name = ev.__class__.__name__

    if name == "RetrievalStartEvent":
        return {
            "type": "tool_call",
            "tool_name": "book_search",
            "tool_kwargs": {"query": getattr(ev, "query", "")},
        }
    if name == "RetrievalDoneEvent":
        return {
            "type": "tool_result",
            "tool_name": "book_search",
            "preview": f"检索到 {getattr(ev, 'count', 0)} 段",
        }
    if name == "AnswerDeltaEvent":
        delta = getattr(ev, "delta", "")
        if delta:
            return {"type": "delta", "data": delta}
    return None
