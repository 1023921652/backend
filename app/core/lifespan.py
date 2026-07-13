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

    logger.info("building agent singleton at startup...")

    try:
        from app.rag.document_rag.tools import (
            rag_decomposed_search,
            rag_simple_search,
        )
        rag_tools = [rag_simple_search, rag_decomposed_search]
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

    # 这里不主动关闭 state_redis_client，进程退出时由 OS 回收
