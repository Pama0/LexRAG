"""数据访问层：会话与消息 CRUD"""
import json
import uuid
from datetime import datetime

from sqlalchemy import delete, desc, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.agent.source_context import SourceRef
from core.persistence.db import MessageRow, SessionRow

TITLE_MAX_CHARS = 20  # 自动标题截取的字符数


def _truncate_title(text: str) -> str:
    text = text.strip().replace("\n", " ")
    return text[:TITLE_MAX_CHARS] + ("…" if len(text) > TITLE_MAX_CHARS else "")


# ==================== Session CRUD ====================

async def create_session(db: AsyncSession, title: str = "新会话") -> SessionRow:
    row = SessionRow(id=str(uuid.uuid4()), title=title)
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def get_session(db: AsyncSession, session_id: str) -> SessionRow | None:
    return await db.get(SessionRow, session_id)


async def list_sessions(db: AsyncSession) -> list[SessionRow]:
    """按最近更新倒序，附消息数"""
    stmt = select(SessionRow).order_by(desc(SessionRow.updated_at))
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def rename_session(db: AsyncSession, session_id: str, title: str) -> bool:
    stmt = (
        update(SessionRow)
        .where(SessionRow.id == session_id)
        .values(title=title.strip()[:200], updated_at=datetime.utcnow())
    )
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0


async def update_summary(
    db: AsyncSession, session_id: str, summary: str, summarized_upto_id: int
) -> bool:
    """更新会话的滚动摘要 + 已摘要水位（上下文压缩用）。"""
    stmt = (
        update(SessionRow)
        .where(SessionRow.id == session_id)
        .values(summary=summary, summarized_upto_id=summarized_upto_id)
    )
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0


async def delete_session(db: AsyncSession, session_id: str) -> bool:
    stmt = delete(SessionRow).where(SessionRow.id == session_id)
    res = await db.execute(stmt)
    await db.commit()
    return res.rowcount > 0


async def count_messages(db: AsyncSession, session_id: str) -> int:
    stmt = select(func.count(MessageRow.id)).where(MessageRow.session_id == session_id)
    return (await db.execute(stmt)).scalar_one()


# ==================== Message CRUD ====================

async def list_messages(db: AsyncSession, session_id: str) -> list[MessageRow]:
    stmt = (
        select(MessageRow)
        .where(MessageRow.session_id == session_id)
        .order_by(MessageRow.id)
    )
    res = await db.execute(stmt)
    return list(res.scalars().all())


async def add_message(
    db: AsyncSession,
    session_id: str,
    role: str,
    content: str,
    sources: list[SourceRef] | None = None,
    auto_title_from_first: bool = False,
) -> MessageRow:
    """写入一条消息；可选：如果这是第一条 user 消息，用 content 自动设标题"""
    sources_json = None
    if sources:
        sources_json = json.dumps(
            [s.model_dump() for s in sources], ensure_ascii=False
        )

    msg = MessageRow(
        session_id=session_id,
        role=role,
        content=content,
        sources_json=sources_json,
    )
    db.add(msg)

    # 同步 session.updated_at；如需自动标题，set title
    session_row = await db.get(SessionRow, session_id)
    if session_row is not None:
        session_row.updated_at = datetime.utcnow()
        if auto_title_from_first and session_row.title in ("新会话", "", None):
            session_row.title = _truncate_title(content)

    await db.commit()
    await db.refresh(msg)
    return msg


def parse_sources(row: MessageRow) -> list[dict]:
    """把 sources_json 还原为前端可用的 dict 列表"""
    if not row.sources_json:
        return []
    try:
        return json.loads(row.sources_json)
    except json.JSONDecodeError:
        return []
