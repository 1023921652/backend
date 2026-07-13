"""Agent 中间件：清洗消息历史中的孤儿 tool_calls。

工具改名 / 流式中断 后，旧 thread 历史里可能存在
AI 消息携带 tool_calls 但缺少匹配的 ToolMessage 响应。
DeepSeek/OpenAI 校验消息历史时会返回 400：
"An assistant message with 'tool_calls' must be followed by tool messages..."

OrphanToolCallSanitizerMiddleware 在每次 model 调用前剔除这种孤儿序列，
让 agent 对工具改名 / 流中断 具备容错能力。
"""
from __future__ import annotations

from typing import Callable

from langchain.agents.factory import ModelRequest
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage


def _sanitize_messages(messages: list) -> list:
    """剔除孤儿 tool_calls。

    策略：AI 消息的 tool_calls 中只要有一个 id 没有匹配 ToolMessage 响应，
    整条 AI 消息丢弃，并记录其所有 tool_call_id，后续对应 ToolMessage 也跳过。
    """
    if not messages:
        return []

    responded_ids = {
        m.tool_call_id for m in messages if isinstance(m, ToolMessage)
    }

    cleaned: list = []
    skip_tool_msg_ids: set[str] = set()

    for msg in messages:
        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            orphaned = any(
                tc.get("id") not in responded_ids for tc in msg.tool_calls
            )
            if orphaned:
                skip_tool_msg_ids.update(tc.get("id") for tc in msg.tool_calls)
                continue
        if (
            isinstance(msg, ToolMessage)
            and msg.tool_call_id in skip_tool_msg_ids
        ):
            continue
        cleaned.append(msg)

    return cleaned


class OrphanToolCallSanitizerMiddleware(AgentMiddleware):
    """在 model 调用前清洗孤儿 tool_calls。"""

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], object],
    ) -> object:
        request.messages = _sanitize_messages(request.messages)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], object],
    ) -> object:
        request.messages = _sanitize_messages(request.messages)
        return await handler(request)
