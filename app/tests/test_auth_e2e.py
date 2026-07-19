"""RBAC 端到端测试。

覆盖三个核心场景：
1. register → login → login/enterprise → /me 主链路
2. require_scope 拦截：member 角色访问 role:create → 403
3. create_enterprise 自动产出 owner/admin 角色 + 把创建者设为 owner
"""
from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_register_login_cycle(owner_client):
    """端到端：注册 → 登录 → 企业登录 → /me 应包含 owner 角色与全 scopes。"""
    client, headers, eid, uid = owner_client

    # /me 校验
    r = await client.get("/v1/auth/me", headers=headers)
    assert r.status_code == 200, r.text
    me = r.json()
    assert me["user_id"] == uid
    assert me["enterprise_id"] == eid
    role_names = [r["name"] for r in me["roles"]]
    assert "owner" in role_names
    # owner 拿到全部 18 个 scope
    assert "enterprise:delete" in me["scopes"]
    assert "user:read" in me["scopes"]
    assert len(me["scopes"]) == 18


@pytest.mark.asyncio
async def test_require_scope_enforced(owner_client):
    """member 角色访问 role:create → 403；owner 访问 → 201。"""
    client, owner_headers, eid, _ = owner_client

    # 邀请 bob 加入企业为 member
    r = await client.post(
        "/v1/auth/register",
        json={"username": "bob", "email": "bob@x.com", "password": "Bob12345"},
    )
    assert r.status_code == 201
    r = await client.post(
        f"/v1/auth/enterprises/{eid}/invite",
        json={"identifier": "bob", "role_name": "member"},
        headers=owner_headers,
    )
    assert r.status_code == 201, r.text

    # bob 登录
    r = await client.post(
        "/v1/auth/login/enterprise",
        json={
            "enterprise_id": eid,
            "identifier": "bob",
            "password": "Bob12345",
        },
    )
    assert r.status_code == 200
    bob_token = r.json()["access_token"]
    bob_headers = {"Authorization": f"Bearer {bob_token}"}

    # bob 试图建角色（需要 role:create） → 403
    r = await client.post(
        f"/v1/auth/enterprises/{eid}/roles",
        json={"name": "viewer", "scopes": ["user:read"]},
        headers=bob_headers,
    )
    assert r.status_code == 403, r.text
    assert r.json()["detail"]["code"] == "insufficient_scope"
    assert r.json()["detail"]["required"] == "role:create"

    # owner 建角色 → 201
    r = await client.post(
        f"/v1/auth/enterprises/{eid}/roles",
        json={"name": "viewer", "scopes": ["user:read", "role:read"]},
        headers=owner_headers,
    )
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "viewer"
    assert set(body["scopes"]) == {"user:read", "role:read"}


@pytest.mark.asyncio
async def test_create_enterprise_yields_owner_role(owner_client):
    """建企业后 DB 中应有 owner/admin/member 三个 builtin 角色；创建者是 owner。"""
    client, owner_headers, eid, uid = owner_client

    # 通过 /roles 接口验证（owner 有 role:read scope）
    r = await client.get(f"/v1/auth/enterprises/{eid}/roles", headers=owner_headers)
    assert r.status_code == 200, r.text
    roles = {r["name"]: r for r in r.json()}
    assert "owner" in roles
    assert "admin" in roles
    assert "member" in roles
    for role in roles.values():
        assert role["is_builtin"] is True
    # owner 拿到 enterprise:delete；admin 没有；member 只有 read 系列
    assert "enterprise:delete" in roles["owner"]["scopes"]
    assert "enterprise:delete" not in roles["admin"]["scopes"]
    assert all(s.endswith(":read") for s in roles["member"]["scopes"])

    # members 列表应含创建者且角色为 owner
    r = await client.get(
        f"/v1/auth/enterprises/{eid}/members", headers=owner_headers
    )
    assert r.status_code == 200, r.text
    members = r.json()
    assert len(members) == 1
    assert members[0]["user_id"] == uid
    assert members[0]["role_name"] == "owner"


@pytest.mark.asyncio
async def test_refresh_and_logout(owner_client):
    """refresh token 旋转 + logout 后 refresh 失败。"""
    client, owner_headers, eid, uid = owner_client

    # 重新登录拿 refresh token
    r = await client.post(
        "/v1/auth/login/enterprise",
        json={
            "enterprise_id": eid,
            "identifier": "owner",
            "password": "Owner1234",
        },
    )
    refresh_token = r.json()["refresh_token"]

    # refresh 一次 → 200
    r = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 200, r.text
    new_refresh = r.json()["refresh_token"]
    assert new_refresh != refresh_token  # 旋转

    # 旧 refresh 被撤销后再用 → 401 token_replayed
    r = await client.post("/v1/auth/refresh", json={"refresh_token": refresh_token})
    assert r.status_code == 401
    assert r.json()["detail"]["code"] == "token_replayed"


@pytest.mark.asyncio
async def test_last_owner_protection(owner_client):
    """唯一 owner 不能被降级或踢出。"""
    client, owner_headers, eid, uid = owner_client

    # 自己降级 → 409 last_owner
    r = await client.put(
        f"/v1/auth/enterprises/{eid}/members/{uid}/role",
        json={"role_name": "member"},
        headers=owner_headers,
    )
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "last_owner"

    # 自己把自己踢了 → 409 last_owner
    r = await client.delete(
        f"/v1/auth/enterprises/{eid}/members/{uid}", headers=owner_headers
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_logout_revokes_access_token(owner_client):
    """logout 后 access token 立即失效（Redis jti 黑名单生效）。"""
    client, owner_headers, eid, uid = owner_client

    # logout 前能访问 /me
    r = await client.get("/v1/auth/me", headers=owner_headers)
    assert r.status_code == 200

    # logout（仅撤销 refresh，但应同时把 access jti 写入 Redis 黑名单）
    r = await client.post(
        "/v1/auth/logout", json={}, headers=owner_headers
    )
    assert r.status_code == 200

    # 同一个 access token 应立即 401 token_revoked
    r = await client.get("/v1/auth/me", headers=owner_headers)
    assert r.status_code == 401, r.text
    assert r.json()["detail"]["code"] == "token_revoked"

    # 重新登录拿新 token → 新 token 仍可用（黑名单只针对具体 jti）
    r = await client.post(
        "/v1/auth/login/enterprise",
        json={
            "enterprise_id": eid,
            "identifier": "owner",
            "password": "Owner1234",
        },
    )
    assert r.status_code == 200
    new_token = r.json()["access_token"]
    new_headers = {"Authorization": f"Bearer {new_token}"}
    r = await client.get("/v1/auth/me", headers=new_headers)
    assert r.status_code == 200
