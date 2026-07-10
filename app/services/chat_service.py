"""把 OpenAI 请求映射到 LangChain Agent 调用。

两种路径：
1. OpenWebUI task 请求（生成 follow_ups / tags / 标题等）→ 绕过 agent，
   直接 deepseek_llm.ainvoke / astream，不写 Redis checkpoint。
2. 普通对话 → 走 agent，且只取最后一条消息（OpenWebUI 会把全量历史
   一起发，但 agent 的 checkpoint 已经维护了历史，重复传会污染 + 重复计费）。
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.schemas.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
)

logger = logging.getLogger(__name__)

# ==========================================
# OpenWebUI task 请求识别
# ==========================================
_OPENWEBUI_TASK_MARKERS = (
    "### Task:",
    "<chat_history>",
)


def is_openwebui_task(messages) -> bool:
    """识别 OpenWebUI 的辅助请求（follow_ups / tags / 标题生成等）。

    特征：最后一条消息 content 包含 "### Task:" 或 "<chat_history>"。
    这些请求是 OpenWebUI 自动发起的辅助任务，不应进 agent、不应写 checkpoint。

    诊断日志：每次调用都打印 matched marker 与 content 前 200 字，
    便于排查 markers 是否覆盖 OpenWebUI 实际 prompt（不同版本/不同 task 模板）。
    """
    if not messages:
        return False
    last = messages[-1]
    content = (last.content or "")
    matched = next(
        (m for m in _OPENWEBUI_TASK_MARKERS if m in content),
        None,
    )
    logger.info(
        "task detect: matched=%s role=%s stream_likely=%s content_preview=%r",
        matched or "none",
        last.role,
        "unknown",
        content[:200],
    )
    return matched is not None


def _take_last(openai_msgs) -> list:
    """只保留最后一条消息（无论 role）。

    - 普通对话：OpenWebUI 全量历史与 checkpoint 重复，只传新消息给 agent
    - task 请求：task prompt 自带 chat_history 嵌入，前面消息多余
    """
    if not openai_msgs:
        return []
    return [openai_msgs[-1]]


# ==========================================
# 公共工具
# ==========================================
_ROLE_MAP: dict[str, type[BaseMessage]] = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}


def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _map_messages(openai_msgs) -> list[BaseMessage]:
    mapped: list[BaseMessage] = []
    for m in openai_msgs:
        cls = _ROLE_MAP.get(m.role)
        if cls is None:
            raise ValueError(f"Unsupported role: {m.role}")
        mapped.append(cls(content=m.content or ""))
    return mapped


def _extract_content(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
        return "".join(parts)
    return str(content)


def _build_response(req: ChatCompletionRequest, content: str) -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id=_gen_id(),
        created=int(time.time()),
        model=req.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=content),
                finish_reason="stop",
            )
        ],
    )


def _stream_payload(chunk_id: str, created: int, model: str, delta: dict, finish_reason) -> str:
    body = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return json.dumps(body, ensure_ascii=False)


# ==========================================
# 非流式入口
# ==========================================
async def nonstream_chat(agent, llm, req: ChatCompletionRequest, thread_id: str) -> ChatCompletionResponse:
    """根据请求类型分流：task → 直接 LLM；普通对话 → agent（只传最后一条）。

    thread_id 仍传给 agent 的 config（普通对话路径写 checkpoint 用），
    但不再回传到响应体（OpenAI 标准 ChatCompletionResponse 无 user 字段）。
    实际使用的 thread_id 由上层通过响应头 X-Thread-Id 回传。
    """
    if is_openwebui_task(req.messages):
        # task 请求：自包含 prompt，不写 checkpoint
        lc_msgs = _map_messages(_take_last(req.messages))
        result = await llm.ainvoke(lc_msgs)
        content = _extract_content(result)
        logger.info(
            "task nonstream response: n_lc_msgs=%d content_preview=%r",
            len(lc_msgs),
            content[:500],
        )
        return _build_response(req, content)

    # 普通对话：只传最后一条新消息，让 agent 用 checkpoint 里的历史
    lc_msgs = _map_messages(_take_last(req.messages))
    result = await agent.ainvoke(
        {"messages": lc_msgs},
        config={"configurable": {"thread_id": thread_id}},
    )
    final_messages: list[BaseMessage] = result.get("messages", [])
    final: BaseMessage = final_messages[-1] if final_messages else AIMessage(content="")
    return _build_response(req, _extract_content(final))


# ==========================================
# 流式入口
# ==========================================
async def stream_chat(agent, llm, req: ChatCompletionRequest, thread_id: str) -> AsyncIterator[str]:
    """流式分流。"""
    chunk_id = _gen_id()
    created = int(time.time())
    model = req.model
    first = True

    if is_openwebui_task(req.messages):
        # task → 直接 LLM 流式
        lc_msgs = _map_messages(_take_last(req.messages))
        parts: list[str] = []
        async for chunk in llm.astream(lc_msgs):
            text = _extract_content(chunk)
            if not text:
                continue
            parts.append(text)
            if first:
                delta = {"role": "assistant", "content": text}
                first = False
            else:
                delta = {"content": text}
            yield f"data: {_stream_payload(chunk_id, created, model, delta, None)}\n\n"
        logger.info(
            "task stream response: n_lc_msgs=%d total_len=%d content_preview=%r",
            len(lc_msgs),
            sum(len(p) for p in parts),
            "".join(parts)[:500],
        )
    else:
        # 普通对话 → agent 流式
        lc_msgs = _map_messages(_take_last(req.messages))
        async for ev in agent.astream_events(
            {"messages": lc_msgs},
            config={"configurable": {"thread_id": thread_id}},
            version="v2",
        ):
            if ev.get("event") != "on_chat_model_stream":
                continue
            chunk = ev.get("data", {}).get("chunk")
            if chunk is None:
                continue
            text = _extract_content(chunk)
            if not text:
                continue
            if first:
                delta = {"role": "assistant", "content": text}
                first = False
            else:
                delta = {"content": text}
            yield f"data: {_stream_payload(chunk_id, created, model, delta, None)}\n\n"

    # 终止块
    yield f"data: {_stream_payload(chunk_id, created, model, {}, 'stop')}\n\n"
    yield "data: [DONE]\n\n"