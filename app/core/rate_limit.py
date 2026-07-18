"""Rate limit 中间件：基于 slowapi。

策略：
- 按 endpoint 配额：ingest 重操作限速更严，chat 流式宽松
- 单位：每 IP 每分钟
- 多 pod 部署：slowapi 默认 in-memory 不准确，需配 Redis backend；
  本期单 pod 测试足够，多 pod 部署时在 register_rate_limiter 里换 backend

超限响应：slowapi 默认抛 RateLimitExceeded，由 _rate_limit_exceeded_handler 转 OpenAI ErrorResponse 格式 429。
"""
from __future__ import annotations

import logging
import os

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.schemas.openai_types import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


# 配额（每 IP / 分钟）
RATE_LIMIT_INGEST_PER_MIN = _env_int("RATE_LIMIT_INGEST_PER_MIN", 10)
RATE_LIMIT_SEARCH_PER_MIN = _env_int("RATE_LIMIT_SEARCH_PER_MIN", 60)
RATE_LIMIT_CHAT_PER_MIN = _env_int("RATE_LIMIT_CHAT_PER_MIN", 120)
RATE_LIMIT_DEFAULT_PER_MIN = _env_int("RATE_LIMIT_DEFAULT_PER_MIN", 240)

# 单例 Limiter；keying 按 IP（前置代理后取 X-Forwarded-For 需在 slowapi.util 改）
# 多 pod 部署时通过 storage_uri='redis://...' 切 Redis backend
_REDIS_URI = os.getenv("RATE_LIMIT_REDIS_URI")
_storage_uri = _REDIS_URI or "memory://"
# key_func=get_remote_address 规定以客户端 IP 地址作为限制标识（每个 IP 独立计算配额）。
# storage_uri 指定计数器存储位置。
limiter = Limiter(key_func=get_remote_address, storage_uri=_storage_uri)
logger.info(
    "rate limiter init: storage=%s ingest=%d/min search=%d/min chat=%d/min default=%d/min",
    "redis" if _REDIS_URI else "memory",
    RATE_LIMIT_INGEST_PER_MIN,
    RATE_LIMIT_SEARCH_PER_MIN,
    RATE_LIMIT_CHAT_PER_MIN,
    RATE_LIMIT_DEFAULT_PER_MIN,
)


def openai_rate_limit_handler(request, exc: RateLimitExceeded):
    """把 slowapi 默认 429 响应改成 OpenAI ErrorResponse 格式（与 errors.py 一致）。

    /v1/ 路径返回 {"error": {...}}；其它路径返回 {"detail": ...} 保持 FastAPI 默认。
    slowapi 的 exc.detail 是描述字符串（如 "10 per 1 minute"），不同版本字段
    结构有差异，统一用 str(exc.detail) 兜底。
    """
    from fastapi.responses import JSONResponse

    path = request.url.path
    detail = str(exc.detail) if exc.detail else "rate limit exceeded"
    msg = f"Rate limit exceeded: {detail}"
    retry_after = getattr(exc, "retry_after", None) or 60

    if path.startswith("/v1/"):
        return JSONResponse(
            status_code=429,
            content=ErrorResponse(
                error=ErrorBody(
                    message=msg,
                    type="rate_limit_error",
                    code="rate_limit_exceeded",
                )
            ).model_dump(),
            headers={"Retry-After": str(retry_after)},
        )
    return JSONResponse(
        status_code=429,
        content={"detail": msg},
        headers={"Retry-After": str(retry_after)},
    )


def register_rate_limiter(app) -> None:
    """在 main.py 注册 limiter state + handler。"""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, openai_rate_limit_handler)
