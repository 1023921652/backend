"""OpenAI Chat Completions 协议的 Pydantic 模型（手写子集）。

仅覆盖本项目实际使用的字段。不直接复用 openai SDK 的内部 Pydantic，
原因：跨版本不稳定，且作为 FastAPI response_model 时 OpenAPI schema 生成易失败。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ==========================================
# 请求
# ==========================================
class ChatMessage(BaseModel):
    """单条消息。tool 角色当前不支持（agent tools=[]）。"""
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI Chat Completions 请求体。
    """
    model: str = "deepseek-chat"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    n: Optional[int] = 1
    stop: Optional[list[str] | str] = None
    user: Optional[str] = None
    metadata: Optional[dict] = None


# ==========================================
# 非流式响应
# ==========================================
class ChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: ChoiceMessage
    finish_reason: Optional[str] = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


# ==========================================
# 流式响应（每个 SSE data 行的 payload）
# ==========================================
class ChunkDelta(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChunkChoice(BaseModel):
    index: int = 0
    delta: ChunkDelta = Field(default_factory=ChunkDelta)
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]


# ==========================================
# /v1/models
# ==========================================
class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "deepseek"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


# ==========================================
# 错误
# ==========================================
class ErrorBody(BaseModel):
    message: str
    type: str = "internal_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorBody
