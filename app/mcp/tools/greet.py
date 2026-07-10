# 在你的 mcp/ 目录下，创建带有装饰器函数的 Python 文件。
# mcp/tools/greet.py
from fastmcp.tools import tool

@tool
def greet(name: str) -> str:
    """Greet someone by name."""
    return f"Hello, {name}!"