"""Chat 路由：所有问答走 Agent 模式，历史用 SQLite 持久化"""
import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, SourceRef
from core.agent.source_context import node_to_source_ref
from core.persistence import repositories as repo
from core.persistence.db import get_session
from core.workflow.doc_query_service import DocQueryService
from core.workflow.summarizer import (
    SUMMARY_KEEP_LAST_MSGS,
    SUMMARY_TRIGGER_MSGS,
    fold_summary,
    plan_overflow,
)

logger = logging.getLogger(__name__)


async def _ensure_session(session_id: str | None) -> str:
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
            # 从 DB 加载摘要 + 未摘要的最近历史构造 memory
            memory, history_len = await _load_memory(query_service, session_id)

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
                is_first_in_session=(history_len == 0),
            )
            # 答复已就绪后再压缩（非流式会在触发轮多等一次摘要 LLM）
            await _maybe_compact(query_service, session_id)

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
                memory, history_len = await _load_memory(query_service, session_id)
                is_first = history_len == 0

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
                    # done 已发出（客户端已拿到完整答案）后再压缩，不拖慢回答
                    await _maybe_compact(query_service, session_id)
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

async def _load_memory(query_service, session_id: str):
    """读会话摘要 + 历史，按水位过滤出【未摘要的最近消息】构造 memory。

    已折入摘要的旧消息由 summary 代表（前置进 memory），其余原文照常带入。
    返回 (memory, history_len)；history_len 用于判定是否首条（自动标题）。
    """
    async with get_session() as db:
        sess = await repo.get_session(db, session_id)
        history = await repo.list_messages(db, session_id)
    upto = (sess.summarized_upto_id or 0) if sess else 0
    summary = sess.summary if sess else None
    recent = [m for m in history if m.id > upto]
    return query_service.build_memory(recent, summary=summary), len(history)


async def _maybe_compact(query_service, session_id: str) -> None:
    """答复送出后的会话压缩：未摘要消息超阈值则增量折叠进摘要。

    绝不影响对话：任何异常只记日志。在 per-session 锁内调用，同会话不并发。
    LLM 摘要调用放在两个短 DB 事务【之间】，不长时间占着连接。
    """
    try:
        async with get_session() as db:
            sess = await repo.get_session(db, session_id)
            messages = await repo.list_messages(db, session_id)
        if sess is None:
            return
        overflow, new_upto = plan_overflow(
            messages, sess.summarized_upto_id or 0,
            trigger=SUMMARY_TRIGGER_MSGS, keep_last=SUMMARY_KEEP_LAST_MSGS,
        )
        if overflow is None:
            return
        new_summary = await fold_summary(query_service.llm, sess.summary, overflow)
        async with get_session() as db:
            await repo.update_summary(db, session_id, new_summary, new_upto)
        logger.info(
            "compact: session=%s 折叠 %d 条 → 摘要，水位至 id=%d",
            session_id, len(overflow), new_upto,
        )
    except Exception:
        logger.exception("会话摘要压缩失败（不影响对话）")


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
    - ThinkingStartEvent  → thinking（前端起 spinner + 计时，收到首个 delta 即停）
    - RetrievalStartEvent → tool_call（前端显示"调用检索"）
    - RetrievalDoneEvent  → tool_result（前端据此进入"答案阶段"）
    - AnswerDeltaEvent    → delta（逐 token 流入答案区）
    """
    name = ev.__class__.__name__

    if name == "ThinkingStartEvent":
        return {"type": "thinking"}
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
