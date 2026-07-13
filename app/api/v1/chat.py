"""OpenAI 标准 /v1/chat/completions 与 /v1/models 路由。"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from app.schemas.openai_types import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelList,
    ModelObject,
)
from app.services.chat_service import nonstream_chat, stream_chat
from app.agent.llm import get_llm
router = APIRouter(tags=["openai"])

logger = logging.getLogger("thread_id")

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek-chat")
KNOWN_MODELS = [
    m.strip()
    for m in os.getenv("KNOWN_MODELS", "deepseek-chat,deepseek-reasoner").split(",")
    if m.strip()
]

THREAD_ID_HEADER = "X-Thread-Id"
OPENWEBUI_CHAT_HEADER = "X-OpenWebUI-Chat-Id"
OPENWEBUI_USER_HEADER = "X-OpenWebUI-User-Id"

# header 名小写集合：诊断日志里只记长度不记 value，避免泄露 token/cookie
_SENSITIVE_HEADERS = frozenset({"authorization", "cookie"})


def _content_fingerprint(messages) -> str | None:
    """取首条 user 消息 content 做 sha1 短哈希，作为弱窗口标识。

    用于 OpenWebUI 未传 metadata/chat_id/header 时的兜底。OpenWebUI 把全量
    历史塞进每次请求，同一窗口不同轮次里"首条 user 消息"始终是开场白，
    因此对同一窗口稳定。

    已知 corner case：两个不同窗口首条消息完全相同（含标点/空格）会串话。
    归一化（strip+lower+截 256 字符）能容忍大小写/前后空白差异，但无法解决
    完全相同开场。
    """
    for m in messages or []:
        if getattr(m, "role", None) == "user" and m.content:
            normalized = m.content.strip().lower()[:256]
            return "fp-" + hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:16]
    return None


def get_agent(request: Request):
    """从 app.state 取 lifespan 启动时构建的 agent 单例。
    """
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    return agent





def _resolve_thread_id(req: ChatCompletionRequest, request: Request) -> tuple[str, str]:
    """按优先级解析 LangGraph thread_id，返回 (thread_id, source_name)。

    顺序（命中即返回）：
    1. req.metadata.chat_id            —— 窗口级（OpenWebUI 标准，最理想）
    2. req.metadata.session_id         —— 会话级
    3. header X-OpenWebUI-Chat-Id      —— 窗口级（v0.6.17+ 默认发，需反向代理不剥离）
    4. header X-Thread-Id              —— 客户端显式指定
    5. header X-OpenWebUI-User-Id      —— 用户级（弱）：单用户场景能恢复多轮
    6. 首条 user 消息内容哈希          —— 窗口级（弱）：OpenWebUI 全量历史里首条稳定
    7. str(uuid.uuid4())               —— 无标识兜底（每次新会话）

    阶段 A 诊断结论：本机 OpenWebUI 1-5 全部不发，落到 6 才能恢复多轮记忆。
    阶段 B 启用 5/6 弱标识；corner case 见 _content_fingerprint docstring。

    诊断日志：
    - DEBUG：metadata keys + 所有 header（Authorization/Cookie 仅记长度，长 value 也只记长度）
    - INFO：最终命中的 source 与 value（value 是 thread_id，无敏感性）
    """
    metadata = req.metadata or {}
    candidates = (
        ("metadata.chat_id", metadata.get("chat_id")),
        ("metadata.session_id", metadata.get("session_id")),
        (f"header.{OPENWEBUI_CHAT_HEADER}", request.headers.get(OPENWEBUI_CHAT_HEADER)),
        (f"header.{THREAD_ID_HEADER}", request.headers.get(THREAD_ID_HEADER)),
        (f"header.{OPENWEBUI_USER_HEADER}", request.headers.get(OPENWEBUI_USER_HEADER)),
        ("content-hash", _content_fingerprint(req.messages)),
    )
    for source_name, value in candidates:
        if value:
            thread_id = str(value)
            logger.info("thread_id resolved: source=%s value=%s", source_name, thread_id)
            return thread_id, source_name

    thread_id = str(uuid.uuid4())
    logger.info("thread_id resolved: source=uuid value=%s", thread_id)
    return thread_id, "uuid"


@router.post(
    "/v1/chat/completions",
    response_model=ChatCompletionResponse,
    response_model_exclude_none=True,
    responses={
        200: {
            "content": {
                "text/event-stream": {"schema": {"type": "string"}},
            },
        },
    },
)
async def chat_completions(
    request: Request,
    response: Response,
    req: ChatCompletionRequest,
    agent=Depends(get_agent),
    llm=Depends(get_llm),
):
    """OpenAI 兼容的 chat completions 入口。

    - 非流式：返回 ChatCompletionResponse JSON
    - 流式：返回 StreamingResponse（SSE）

    OpenWebUI task 请求（follow_ups/tags/标题生成）走直 LLM 路径，
    不进 agent、不写 Redis checkpoint，避免污染会话历史。

    thread_id 来源优先级见 _resolve_thread_id；opwenwebui那边在请求头中使用 X-Thread-Id
    """
    if req.messages and any(m.role == "tool" for m in req.messages):
        raise HTTPException(status_code=400, detail="tool role messages are not supported")

    thread_id, _source = _resolve_thread_id(req, request)
    # 非流式：注入到默认 Response 的 headers，FastAPI 会合并到最终响应
    response.headers[THREAD_ID_HEADER] = thread_id

    if req.stream:
        return StreamingResponse(
            stream_chat(agent, llm, req, thread_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",# 禁止浏览器和中间代理服务器（如 CDN、反向代理等）对该流式请求的响应进行缓存。
                "X-Accel-Buffering": "no",# 告诉 Nginx（或其它支持该头部的反向代理服务器）关闭响应缓冲区，立即将数据发送给客户端。
                "Connection": "keep-alive",# 指示客户端和中间代理保持当前 TCP 连接为打开状态，不要在发送/接收完初始数据后就关闭连接。
                THREAD_ID_HEADER: thread_id,
            },
        )

    return await nonstream_chat(agent, llm, req, thread_id)


@router.get("/v1/models", response_model=ModelList, response_model_exclude_none=True)
async def list_models():
    """OpenAI 兼容的模型列表。"""
    created = int(time.time())
    return ModelList(
        data=[
            ModelObject(id=name, created=created, owned_by="deepseek")
            for name in KNOWN_MODELS
        ]
    )
