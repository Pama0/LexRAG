"""SQLite 持久化层：会话与消息历史

- 使用 SQLAlchemy 2.0 异步 API
- 文件落在项目根目录 ./bookkb.db
- 启动时自动建表
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncIterator, Optional

from sqlalchemy import DateTime, ForeignKey, String, Text, select
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(PROJECT_ROOT, "bookkb.db")
DATABASE_URL = f"sqlite+aiosqlite:///{DB_PATH}"


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(200), default="新会话")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    # 上下文压缩：远期历史的滚动摘要 + 已折入摘要的最大消息 id（水位）。
    # 见 core/workflow/summarizer.py（持久化增量摘要）。
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True, default=None)
    summarized_upto_id: Mapped[int] = mapped_column(default=0)

    messages: Mapped[list["MessageRow"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="MessageRow.id",
    )


class MessageRow(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    # sources 序列化为 JSON 字符串；为简单不引 JSON 列类型
    sources_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    session: Mapped[SessionRow] = relationship(back_populates="messages")


# 引擎和 session 工厂（模块级单例）
_engine = create_async_engine(DATABASE_URL, echo=False, future=True)
async_session_factory = async_sessionmaker(_engine, expire_on_commit=False)


def _migrate_add_columns(conn) -> None:
    """给已有 sessions 表补新列（create_all 只建缺表、不加列）。幂等：缺啥补啥。"""
    existing = {row[1] for row in conn.exec_driver_sql(
        "PRAGMA table_info(sessions)").fetchall()}
    if "summary" not in existing:
        conn.exec_driver_sql("ALTER TABLE sessions ADD COLUMN summary TEXT")
    if "summarized_upto_id" not in existing:
        conn.exec_driver_sql(
            "ALTER TABLE sessions ADD COLUMN summarized_upto_id INTEGER NOT NULL DEFAULT 0"
        )


async def init_db() -> None:
    """启动时调用：创建所有表（不存在才建）+ 给已有表补列"""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.run_sync(_migrate_add_columns)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """便利 context manager"""
    async with async_session_factory() as session:
        yield session
