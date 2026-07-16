"""FastAPI 应用入口。

保留原有 /items/{id} 与 /items/ 两条 demo 路由（含 X-Token 校验、内联 fake_db）。
新增 OpenAI 标准 /v1/chat/completions（流式 + 非流式）和 /v1/models 路由。
"""
# 必须最先：把 .env 加载到 os.environ，再让 setup_logging / 各模块读 env 生效
from dotenv import load_dotenv

load_dotenv(".env")

import uvicorn
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.v1.chat import router as chat_router
from app.api.v1.contextual_rag import router as contextual_rag_router
from app.api.v1.rag import router as rag_router
from app.core.errors import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.logging import setup_logging
from app.core.middleware import RequestIdMiddleware

# 启动前先初始化日志（此时 LOG_DIR / LOG_LEVEL_THREAD 等已就绪）
setup_logging()

app = FastAPI(
    title="OpenAI-Compatible Agent Service",
    description="把 LangChain Agent 暴露为 OpenAI 标准 /v1/chat/completions 接口",
    version="0.1.0",
    lifespan=lifespan,
)

# ==========================================
# 中间件（顺序：最后添加的最先执行）
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # "*" + credentials 不合法
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestIdMiddleware)

# ==========================================
# 异常处理（按路径前缀分流错误格式）
# ==========================================
register_exception_handlers(app)

# ==========================================
# OpenAI 兼容路由
# ==========================================
app.include_router(chat_router)
app.include_router(rag_router)
app.include_router(contextual_rag_router)


# ==========================================
# 原 /items demo 路由（保留，测试 test_main.py 覆盖）
# ==========================================
fake_secret_token = "coneofsilence"

fake_db = {
    "foo": {"id": "foo", "title": "Foo", "description": "There goes my hero"},
    "bar": {"id": "bar", "title": "Bar", "description": "The bartenders"},
}


class Item(BaseModel):
    id: str
    title: str
    description: str | None = None


@app.get("/items/{item_id}", response_model=Item)
async def read_main(item_id: str, x_token: Annotated[str, Header()]):
    if x_token != fake_secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item_id not in fake_db:
        raise HTTPException(status_code=404, detail="Item not found")
    return fake_db[item_id]


@app.post("/items/")
async def create_item(item: Item, x_token: Annotated[str, Header()]) -> Item:
    if x_token != fake_secret_token:
        raise HTTPException(status_code=400, detail="Invalid X-Token header")
    if item.id in fake_db:
        raise HTTPException(status_code=409, detail="Item already exists")
    fake_db[item.id] = item.model_dump()
    return item


@app.get("/health", tags=["health"])
async def health():
    """Liveness probe：进程能响应即 200。

    不检查任何外部依赖（Redis / Milvus / LLM），避免外部抖动导致 pod 被重启。
    k8s livenessProbe 用这个；失败 → 重启 pod。
    """
    return {"status": "ok"}


@app.get("/ready", tags=["health"])
async def ready():
    """Readiness probe：agent 单例构造完成才 200，否则 503。

    只校验本 pod 的 lifespan 是否走完。不 ping Redis/Milvus：它们 down 时
    所有 pod 会同时 not-ready，反而造成全局不可用。外部依赖故障让具体请求
    自己报错（/v1/rag/* 走 503，chat 走错误 chunk 或 5xx），pod 仍接流量。
    k8s readinessProbe 用这个；失败 → 从 Service 摘除但不重启。
    """
    if getattr(app.state, "agent", None) is None:
        raise HTTPException(status_code=503, detail="agent not initialized")
    return {"status": "ready"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
