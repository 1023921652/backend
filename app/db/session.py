"""异步 SQLAlchemy 引擎与 session 工厂。

- engine：连接池 size=10，pool_recycle=3600s（MySQL 默认 wait_timeout 28800s，留余量）
- async_sessionmaker：每请求一个 AsyncSession
- get_session：FastAPI 依赖；yield session 后 close
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_recycle=3600,
    pool_pre_ping=True,
    echo=settings.db_echo,
)

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI 依赖：每请求一个 AsyncSession。"""
    async with async_session_factory() as session:
        yield session
