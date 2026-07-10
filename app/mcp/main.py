"""MCP server 入口。

- 用 FileSystemProvider 扫描 tools/、prompts/、resources/ 三个子目录
- 不能扫整个 app/mcp/ 根：会把 main.py 自身当工具模块 import → 重新触发
  FastMCP(providers=...) → 再次扫描 → 递归爆栈
- `python -m app.mcp.main` 作为 stdio 子进程启动，供 MultiServerMCPClient 连接
"""
from pathlib import Path

from fastmcp import FastMCP
from fastmcp.server.providers import FileSystemProvider

_ROOT = Path(__file__).parent

mcp = FastMCP(
    "McpServer",
    providers=[
        FileSystemProvider(_ROOT / "tools"),
        FileSystemProvider(_ROOT / "prompts"),
        FileSystemProvider(_ROOT / "resources"),
    ],
)


if __name__ == "__main__":
    # stdio transport：stdin/stdout 走 JSON-RPC，stderr 仍可 print 调试
    mcp.run(transport="stdio")
