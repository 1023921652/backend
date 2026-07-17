"""全局异常处理：按路径前缀分流错误格式。

/v1/chat/* 与 /v1/rag/* 与 /v1/contextual_rag/* → OpenAI 错误格式 {"error": {...}}
/v1/auth/* 与其余路径 → FastAPI 默认 {"detail": ...}（保留 dict 结构，便于客户端按 code 分支）
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.schemas.openai_types import ErrorBody, ErrorResponse

logger = logging.getLogger(__name__)


_OPENAI_PREFIXES = ("/v1/chat", "/v1/models", "/v1/rag", "/v1/contextual_rag")


def _is_openai_path(path: str) -> bool:
    return any(path.startswith(p) for p in _OPENAI_PREFIXES)


def _openai_error(status: int, message: str, etype: str, code: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content=ErrorResponse(error=ErrorBody(message=message, type=etype, code=code)).model_dump(),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        # FastAPI 的 HTTPException 是 StarletteHTTPException 的子类，一并覆盖
        if _is_openai_path(request.url.path):
            return _openai_error(
                status=exc.status_code,
                message=str(exc.detail),
                etype="api_error" if exc.status_code != 404 else "not_found",
                code=str(exc.status_code),
            )
        # 非 /v1/ 路径走 FastAPI 默认格式 {"detail": ...}
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        if _is_openai_path(request.url.path):
            return _openai_error(
                status=422,
                message=f"Invalid request body: {exc.errors()}",
                etype="invalid_request_error",
                code="invalid_request",
            )
        # 非 /v1/ 走 FastAPI 默认 422 格式
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("unhandled exception on %s", request.url.path)
        if _is_openai_path(request.url.path):
            return _openai_error(
                status=500,
                message="Internal server error",
                etype="internal_error",
                code="internal_error",
            )
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
