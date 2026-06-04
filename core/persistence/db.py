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


async def init_db() -> None:
    """启动时调用：创建所有表（不存在才建）"""
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """便利 context manager"""
    async with async_session_factory() as session:
        yield session
