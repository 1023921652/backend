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

import openai
from fastapi import HTTPException
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

    多模态兼容：last.content 可能是 list[dict]（含 image_url 等），先经
    content_to_text 归一化再做 marker 匹配，避免 list 无法做 `in` / 切片。
    """
    if not messages:
        return False
    last = messages[-1]
    content_text = content_to_text(last.content)
    matched = next(
        (m for m in _OPENWEBUI_TASK_MARKERS if m in content_text),
        None,
    )
    logger.info(
        "task detect: matched=%s role=%s stream_likely=%s content_preview=%r",
        matched or "none",
        last.role,
        "unknown",
        content_text[:200],
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


def content_to_text(content) -> str:
    """把 OpenAI 多模态 content（str | list[dict]）归一化为纯文本。

    list 形态时只拼接 type=text 的 part，丢弃 image_url / audio 等多模态 part。

    使用场景：
    - 日志 / 诊断（不能把 base64 图片打到日志里）
    - thread_id fingerprint（取首条 user 消息文本做 sha1）
    - system / assistant 消息（LangChain 这两类消息 content 期望 str）
    - OpenWebUI task marker 识别
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and item.get("type") == "text":
            parts.append(item.get("text", ""))
    return "".join(parts)


def _content_has_multimodal(content) -> bool:
    """判断 content 是否含非 text 的多模态 part（如 image_url / input_audio）。

    仅 user 消息且本函数返回 True 时，才把 list 透传给 LangChain；其余情况
    一律走 content_to_text 归一化（避免 system/assistant 收到 list 引发 LLM 400）。
    """
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") not in (None, "text")
        for item in content
    )


def _gen_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _openai_error_status(exc: openai.APIError) -> int:
    """openai.APIError → 合适的 HTTP status。5xx 统一降为 503（服务不可用）。"""
    status = getattr(exc, "status_code", None) or 500
    if status >= 500:
        return 503
    return int(status)


def _openai_error_detail(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _stream_error_chunk(chunk_id: str, created: int, model: str, first: bool, exc: Exception) -> str:
    """构造一个 SSE error chunk，让 OpenWebUI 在 UI 展示错误文本而非连接断开。"""
    err_text = f"[stream error] {_openai_error_detail(exc)}"
    delta = (
        {"role": "assistant", "content": err_text}
        if first
        else {"content": err_text}
    )
    return f"data: {_stream_payload(chunk_id, created, model, delta, None)}\n\n"


def _map_messages(openai_msgs) -> list[BaseMessage]:
    """OpenAI ChatMessage → LangChain BaseMessage。

    多模态处理：
    - user 消息含 image_url 等非 text part → 保留 list 结构透传给 LangChain
      HumanMessage（LangChain 接受同形态 list）。当前若底层 LLM 是纯文本模型
      （如 deepseek-chat），LLM 调用阶段会报错——这是预期行为，提示需要切到
      vision 模型。
    - 其余情况（system / assistant，或 user 仅含 text）→ 用 content_to_text
      归一化为字符串。
    """
    mapped: list[BaseMessage] = []
    for m in openai_msgs:
        cls = _ROLE_MAP.get(m.role)
        if cls is None:
            raise ValueError(f"Unsupported role: {m.role}")
        if cls is HumanMessage and _content_has_multimodal(m.content):
            mapped.append(cls(content=m.content))
        else:
            mapped.append(cls(content=content_to_text(m.content) or ""))
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

    LLM 调用异常（openai.APIError：鉴权 / 限流 / bad request）转成 HTTPException，
    由 app/core/errors.py 的全局 handler 包成 OpenAI ErrorResponse 返回。
    """
    if is_openwebui_task(req.messages):
        # task 请求：自包含 prompt，不写 checkpoint
        lc_msgs = _map_messages(_take_last(req.messages))
        try:
            result = await llm.ainvoke(lc_msgs)
        except openai.APIError as e:
            logger.warning("task nonstream LLM error: %s", _openai_error_detail(e))
            raise HTTPException(
                status_code=_openai_error_status(e),
                detail=_openai_error_detail(e),
            )
        content = _extract_content(result)
        logger.info(
            "task nonstream response: n_lc_msgs=%d content_preview=%r",
            len(lc_msgs),
            content[:500],
        )
        return _build_response(req, content)

    # 普通对话：只传最后一条新消息，让 agent 用 checkpoint 里的历史
    lc_msgs = _map_messages(_take_last(req.messages))
    try:
        result = await agent.ainvoke(
            {"messages": lc_msgs},
            config={"configurable": {"thread_id": thread_id}},
        )
    except openai.APIError as e:
        logger.warning("agent nonstream LLM error: %s", _openai_error_detail(e))
        raise HTTPException(
            status_code=_openai_error_status(e),
            detail=_openai_error_detail(e),
        )
    final_messages: list[BaseMessage] = result.get("messages", [])
    final: BaseMessage = final_messages[-1] if final_messages else AIMessage(content="")
    return _build_response(req, _extract_content(final))


# ==========================================
# 流式入口
# ==========================================
async def stream_chat(agent, llm, req: ChatCompletionRequest, thread_id: str) -> AsyncIterator[str]:
    """流式分流。

    LLM 调用异常（openai.APIError / 其它未预期异常）在流式中段抛出时，
    响应已经 200 + headers 发出，无法改 HTTP status。改用一个 SSE error chunk
    告知客户端错误内容，再正常 stop + [DONE] 收尾，避免连接被 RST。
    OpenWebUI 会把 [stream error] ... 作为 assistant 消息文本渲染。
    """
    chunk_id = _gen_id()
    created = int(time.time())
    model = req.model
    first = True
    # 此处，如果openwebui的task（RAG任务）是流式的，也交给agent处理，title，tag，追加问题都不是流式的
    # if is_openwebui_task(req.messages):
    try:
        if False:
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
    except openai.APIError as e:
        logger.warning("stream_chat LLM error: %s", _openai_error_detail(e))
        yield _stream_error_chunk(chunk_id, created, model, first, e)
    except Exception as e:
        # 兜底：LangChain 偶尔会把底层异常包成自定义类型，仍走优雅收尾
        logger.exception("stream_chat unexpected error")
        yield _stream_error_chunk(chunk_id, created, model, first, e)

    # 终止块
    yield f"data: {_stream_payload(chunk_id, created, model, {}, 'stop')}\n\n"
    yield "data: [DONE]\n\n"