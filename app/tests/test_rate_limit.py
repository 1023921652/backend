"""限流测试。

利用 ingest 空请求短路特性（不依赖 milvus），反复打 endpoint 验证 429。
slowapi 默认 in-memory storage，计数器按 IP（TestClient 都来自同一 testclient IP）。
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_ingest_rate_limit_triggers_429():
    """/v1/rag/documents 限流：超过 RATE_LIMIT_INGEST_PER_MIN 后返回 429。"""
    # 读当前配置（.env 默认 10/min）
    from app.core.rate_limit import RATE_LIMIT_INGEST_PER_MIN

    # 多打几次确保超限（额外 5 次冗余）
    over = RATE_LIMIT_INGEST_PER_MIN + 5

    codes: list[int] = []
    for _ in range(over):
        resp = client.post("/v1/rag/documents", json=[])
        codes.append(resp.status_code)

    assert 429 in codes, f"no 429 in sequence: {codes}"
    # 429 响应体应为 OpenAI 错误格式
    rejected = [c for c in codes if c == 429]
    assert len(rejected) >= 5, f"expected ≥5 rejections, got {len(rejected)}"

    # 再取一个 429 响应体验证格式
    resp = client.post("/v1/rag/documents", json=[])
    if resp.status_code == 429:
        body = resp.json()
        assert "error" in body, f"OpenAI ErrorResponse format expected, got {body}"
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert "Retry-After" in resp.headers


def test_health_endpoint_not_rate_limited():
    """/health 不挂 limiter；反复调用应全部 200。"""
    for _ in range(30):
        resp = client.get("/health")
        assert resp.status_code == 200
