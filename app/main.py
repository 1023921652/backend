"""FastAPI 应用入口。

保留原有 /items/{id} 与 /items/ 两条 demo 路由（含 X-Token 校验、内联 fake_db）。
新增 OpenAI 标准 /v1/chat/completions（流式 + 非流式）和 /v1/models 路由。
"""
import uvicorn
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.api.v1.chat import router as chat_router
from app.core.errors import register_exception_handlers
from app.core.lifespan import lifespan
from app.core.logging import setup_logging
from app.core.middleware import RequestIdMiddleware

# 启动前先初始化日志
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
    """健康检查：agent 是否就绪。"""
    return {
        "status": "ok",
        "agent_ready": getattr(app.state, "agent", None) is not None,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
