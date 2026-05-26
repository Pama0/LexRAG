"""Chat 路由：所有问答走 Agent 模式，历史用 SQLite 持久化"""
import json
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api import repositories as repo
from api.agent_service import AgentService
from api.db import get_session
from api.schemas import ChatRequest, ChatResponse, SourceRef
from core.agent.source_context import begin_collection, get_sources

logger = logging.getLogger(__name__)


def _wrap_message(req: ChatRequest) -> str:
    """把可选 book_title 拼到用户消息里作为提示给 Agent"""
    if req.book_title:
        return f"（请在《{req.book_title}》中查找）{req.message}"
    return req.message


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


def create_chat_router(agent_service: AgentService) -> APIRouter:
    """工厂函数：注入 agent_service 依赖"""
    router = APIRouter(prefix="/api", tags=["chat"])

    @router.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest):
        """非流式问答：Agent 推理完返回完整答案 + 来源"""
        if not req.message or not req.message.strip():
            raise HTTPException(status_code=400, detail="消息不能为空")

        session_id = await _ensure_session(req.session_id)
        logger.info("chat: session=%s msg=%r", session_id, req.message[:50])

        lock = agent_service.get_lock(session_id)
        async with lock:
            # 从 DB 加载历史构造 memory
            async with get_session() as db:
                history = await repo.list_messages(db, session_id)
            memory = agent_service.build_memory(history)

            begin_collection()
            try:
                response = await agent_service.agent.run(
                    user_msg=_wrap_message(req),
                    memory=memory,
                )
            except Exception as e:
                logger.exception("Agent run failed")
                raise HTTPException(status_code=500, detail=f"Agent 执行失败: {e}")

            sources = get_sources()
            answer_text = str(response)

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
            "chat_stream: session=%s msg=%r book=%s",
            session_id, req.message[:50], req.book_title,
        )

        lock = agent_service.get_lock(session_id)

        async def event_generator():
            # 第一条事件告诉前端最终 session_id（可能是后端新建的）
            yield f"data: {json.dumps({'type': 'session', 'session_id': session_id})}\n\n"

            async with lock:
                async with get_session() as db:
                    history = await repo.list_messages(db, session_id)
                memory = agent_service.build_memory(history)
                is_first = len(history) == 0

                begin_collection()
                try:
                    handler = agent_service.agent.run(
                        user_msg=_wrap_message(req),
                        memory=memory,
                    )

                    async for ev in handler.stream_events():
                        payload = _format_event(ev)
                        if payload is not None:
                            yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

                    final = await handler
                    final_text = str(final)
                    logger.info("chat_stream done; final length=%d", len(final_text))

                    sources = get_sources()
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
                    logger.exception("Agent stream failed")
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
    """把 LlamaIndex Agent 事件转为前端可消费的轻量 payload"""
    name = ev.__class__.__name__

    if name == "ToolCall":
        return {
            "type": "tool_call",
            "tool_name": getattr(ev, "tool_name", ""),
            "tool_kwargs": getattr(ev, "tool_kwargs", {}),
        }
    if name == "ToolCallResult":
        result = getattr(ev, "tool_output", None)
        text = str(result)[:500] if result is not None else ""
        return {
            "type": "tool_result",
            "tool_name": getattr(ev, "tool_name", ""),
            "preview": text,
        }
    if name == "AgentStream":
        delta = getattr(ev, "delta", "")
        if delta:
            return {"type": "delta", "data": delta}
    return None
