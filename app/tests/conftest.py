"""RBAC 端到端测试 fixtures。

策略：
- 复用 rbac 库（dev 模式 lifespan 会自动 create_all）
- 每测试前用 pymysql（同步驱动，不走 event loop）清表，避开 aiomysql 跨 loop 连接复用问题
- httpx.AsyncClient + ASGITransport(app=app) 进程内调用，避免端口冲突
"""
from __future__ import annotations

import contextlib

import pymysql
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.db.mysql.base import Base
from app.db.mysql.session import engine
from app.core.config import settings


def _truncate_all() -> None:
    """同步 TRUNCATE 所有 RBAC 表。

    用 pymysql 同步执行：避免 async engine 连接跨 event loop 复用导致的 NoneType send 错误。
    """
    conn = pymysql.connect(
        host="localhost", user="root", password="root", database="rbac", charset="utf8mb4"
    )
    try:
        with conn.cursor() as cur:
            cur.execute("SET FOREIGN_KEY_CHECKS=0")
            for table in Base.metadata.sorted_tables:
                cur.execute(f"TRUNCATE TABLE `{table.name}`")
            cur.execute("SET FOREIGN_KEY_CHECKS=1")
        conn.commit()
    finally:
        conn.close()


import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session", autouse=True)
async def _create_tables():
    """session 开始前确保表存在（ASGITransport 不触发 FastAPI lifespan）。

    必须是 async fixture，与测试共用 session-scoped event loop，
    否则 engine 创建的连接会跨 loop 失效。
    """
    from app.db.mysql.base import Base
    from app.db.mysql.session import engine
    from app.auth import models  # noqa: F401 注册元数据

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest.fixture(autouse=True)
def _reset_db():
    """每个测试前同步清表。"""
    _truncate_all()
    yield


@pytest_asyncio.fixture
async def async_client():
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver", timeout=10) as client:
        yield client


@contextlib.asynccontextmanager
async def _bootstrap_enterprise(creator_user_id: int, name: str, slug: str):
    """service 层直建企业，绕过 API 链路。"""
    from app.auth import service as svc
    from app.db.mysql.session import async_session_factory

    async with async_session_factory() as s:
        async with s.begin():
            ent = await svc.create_enterprise(
                s, creator_user_id=creator_user_id, name=name, slug=slug
            )
        yield ent.id


@pytest_asyncio.fixture
async def owner_client(async_client: AsyncClient):
    """注册 owner 用户 + 建企业 + 企业登录，返回 (client, headers, enterprise_id, user_id)。"""
    r = await async_client.post(
        "/v1/auth/register",
        json={"username": "owner", "email": "owner@x.com", "password": "Owner1234"},
    )
    assert r.status_code == 201, r.text
    uid = r.json()["user_id"]

    async with _bootstrap_enterprise(uid, "Acme Inc", "acme") as eid:
        r = await async_client.post(
            "/v1/auth/login/enterprise",
            json={
                "enterprise_id": eid,
                "identifier": "owner",
                "password": "Owner1234",
            },
        )
        assert r.status_code == 200, r.text
        token = r.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}
        yield async_client, headers, eid, uid
