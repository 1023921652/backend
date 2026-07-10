"""MCP client：用 stdio 子进程启动 app.mcp.main，加载工具并暴露给 agent。

设计要点（langchain-mcp-adapters 0.3+ API）：
- MultiServerMCPClient 不再支持 `async with client`；改用 `async with client.session(name)`
  持久化 stdio 子进程，所有 tool 调用复用此 session
- `tools_context()` 是上层 async context manager：启动 server 子进程 → 加载 tools
  → yield tools → 退出时自动 kill 子进程
- _CallLogger 是 ToolCallInterceptor：agent 调用 MCP 工具时在 stdout 打印调用与结果
- __main__ 块是终端 demo
"""
from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.interceptors import MCPToolCallRequest, MCPToolCallResult
from langchain_mcp_adapters.tools import load_mcp_tools

logger = logging.getLogger(__name__)

_SERVER_KEY = "main-mcp-server"
_SERVER_COMMAND = {
    "command": sys.executable,
    "args": ["-m", "app.mcp.main"],
    "transport": "stdio",
}


class _CallLogger:
    """ToolCallInterceptor：agent 调用 MCP 工具时在 stdout 打印调用与结果。

    print + flush=True 保证 uvicorn 终端即时显示。
    """

    async def __call__(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[MCPToolCallResult]],
    ) -> MCPToolCallResult:
        print(
            f"[MCP] calling tool: server={request.server_name} name={request.name} args={request.args}",
            flush=True,
        )
        try:
            result = await handler(request)
        except Exception as e:
            print(
                f"[MCP] tool '{request.name}' failed: {type(e).__name__}: {e}",
                flush=True,
            )
            raise
        print(f"[MCP] tool '{request.name}' succeeded: {result!r}", flush=True)
        return result


def build_mcp_client() -> MultiServerMCPClient:
    """构造未连接的 client；持久 session 通过 tools_context() 建立。"""
    return MultiServerMCPClient({_SERVER_KEY: _SERVER_COMMAND})


@asynccontextmanager
async def tools_context() -> AsyncIterator[list]:
    """启动 MCP server 子进程 → 加载 tools → yield → 退出时关闭子进程。

    用法：
        async with tools_context() as tools:
            agent = await set_agent(mcp_tools=tools)
            yield
    """
    client = build_mcp_client()
    async with client.session(_SERVER_KEY) as session:
        tools = await load_mcp_tools(
            session,
            server_name=_SERVER_KEY,
            tool_interceptors=[_CallLogger()],
        )
        logger.info(
            "loaded %d mcp tools: %s", len(tools), [t.name for t in tools]
        )
        yield tools



