"""轻量中间件：注入 X-Request-ID 并写入 contextvars，便于日志追踪。"""
from __future__ import annotations

import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.context import request_id_ctx_var

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIdMiddleware(BaseHTTPMiddleware):
    """每个请求生成/透传 X-Request-ID，写入响应头 + contextvars。

    contextvars 让同请求内任何协程（包括 agent / langchain 内部日志）
    都能通过 request_id_ctx_var.get() 拿到当前 ID，配合 logging filter
    自动出现在日志 format 中。
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        req_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = req_id
        token = request_id_ctx_var.set(req_id)
        try:
            response = await call_next(request)
        finally:
            request_id_ctx_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = req_id
        return response
