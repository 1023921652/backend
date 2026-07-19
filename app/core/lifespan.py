"""应用 lifespan：启动时构建 agent 单例（含 MCP 工具），停止时关闭 MCP 子进程。"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """启动 MCP server 子进程 → 加载 tools → 构建带 tools 的 agent。

    MCP 子进程必须在 agent 整个生命周期内存活，所以 tools_context() 包住 yield；
    退出时由 __aexit__ 自动 kill 子进程。

    降级：
    - MCP 启动失败 → 退化为无 mcp tools 的 agent（仍可启动）
    - agent 构建失败 → app.state.agent = None，接口返回 503
    - Redis 异常：set_agent 内部 try/except，返回 checkpointer=None 的无状态 agent
    """
    from app.agent.main import set_agent

    # ============ RBAC DB 引擎初始化 ============
    # dev 模式自动建表；prod 用 alembic upgrade head
    try:
        from app.core.config import settings
        from app.db.mysql.base import Base
        from app.db.mysql.session import engine
        from app.auth import models  # noqa: F401  确保表元数据被注册到 Base.metadata

        if settings.env == "dev":
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        logger.info("rbac db engine ready (env=%s)", settings.env)
    except Exception:
        logger.exception("rbac db init failed; auth endpoints will 500 on demand")

    logger.info("building agent singleton at startup...")

    try:
        from app.rag.document_rag.tools import (
            rag_decomposed_search,
            rag_fulltext_search,
            rag_simple_search,
        )
        rag_tools = [rag_simple_search, rag_decomposed_search, rag_fulltext_search]
    except Exception:
        logger.exception("rag tools import failed; agent will start without them")
        rag_tools = []

    rag_count = len(rag_tools)

    try:
        from app.mcp.client import tools_context

        async with tools_context() as mcp_tools:
            all_tools = list(mcp_tools) + rag_tools
            try:
                app.state.agent = await set_agent(mcp_tools=all_tools)
                logger.info(
                    "agent singleton ready (tools=%d, mcp=%d, rag=%d)",
                    len(all_tools),
                    len(mcp_tools),
                    rag_count,
                )
            except Exception:
                logger.exception("failed to build agent at startup")
                app.state.agent = None
            yield
    except Exception:
        logger.exception(
            "mcp setup failed; starting agent without mcp tools"
        )
        try:
            app.state.agent = await set_agent(mcp_tools=list(rag_tools))
            logger.info(
                "agent singleton ready (no mcp tools, rag=%d)", rag_count
            )
        except Exception:
            logger.exception("agent build failed")
            app.state.agent = None
        yield

    # 关闭 Redis 连接池
    try:
        from app.db.redis.pool import pool
        await pool.aclose()
        logger.info("redis pool closed")
    except Exception:
        logger.exception("redis pool close failed")

    # 关闭 RBAC DB 引擎
    try:
        from app.db.mysql.session import engine
        await engine.dispose()
        logger.info("rbac db engine disposed")
    except Exception:
        logger.exception("rbac db engine dispose failed")
