"""FastAPI 依赖：会话、当前用户、scope 校验。

设计：
- get_current_user：解析 Authorization Bearer，解码 access token，返回 AuthContext
- require_scope(scope)：依赖工厂，校验 ctx.scopes 含目标 scope
- assert_tenant：辅助函数，校验 path 参数 eid 与 token.tenant_id 一致（防跨租户）
- jti 黑名单：access token 是短期 token，撤销通过等待过期（≤30min）；
  长期撤销靠 refresh token 撤销 + 拒绝签发新 access token。生产可加 Redis 黑名单。

通过 fastapi.security.HTTPBearer 暴露 Bearer 安全方案，
Swagger UI 右上角会显示 Authorize 按钮，输入 access_token 后自动带 Authorization 头。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.security import decode_token
from app.db.session import get_session

# Swagger UI 会显示 🔒 + Authorize 按钮；auto_error=True 缺失时自动 403
bearer_scheme = HTTPBearer(auto_error=True)


@dataclass(frozen=True)
class AuthContext:
    user_id: int
    tenant_id: int
    roles: list[str]
    scopes: list[str]
    jti: str
    exp: int  # access token 过期时间戳（秒），用于 logout 时算 Redis TTL


SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials, Depends(bearer_scheme)],
    session: SessionDep,
) -> AuthContext:
    """解析 Authorization: Bearer <token> → AuthContext。"""
    token = creds.credentials

    try:
        payload = decode_token(token)
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail={"code": "invalid_token", "message": f"token invalid: {e}"},
        )

    if payload.get("type") != "access":
        raise HTTPException(
            status_code=401,
            detail={"code": "wrong_token_type", "message": "access token required"},
        )

    try:
        user_id = int(payload["sub"])
        tenant_id = int(payload["tenant_id"])
        roles = list(payload.get("roles", []))
        scopes = list(payload.get("scopes", []))
        jti = str(payload["jti"])
        exp = int(payload["exp"])
    except (KeyError, ValueError) as e:
        raise HTTPException(
            status_code=401,
            detail={"code": "malformed_token", "message": f"missing claims: {e}"},
        )

    # access token jti 黑名单查询（logout 后立即失效，Redis 故障 fail-open）
    from app.auth.token_blocklist import is_access_jti_revoked
    if await is_access_jti_revoked(jti):
        raise HTTPException(
            status_code=401,
            detail={"code": "token_revoked", "message": "access token has been revoked"},
        )

    return AuthContext(
        user_id=user_id,
        tenant_id=tenant_id,
        roles=roles,
        scopes=scopes,
        jti=jti,
        exp=exp,
    )


CurrentUser = Annotated[AuthContext, Depends(get_current_user)]


def require_scope(scope: str):
    """依赖工厂：校验 ctx.scopes 含目标 scope。

    用法：
        @router.get("/...", dependencies=[Depends(require_scope("user:read"))])
        或
        def handler(ctx: AuthContext = Depends(require_scope("user:read"))): ...
    """
    async def _dep(ctx: CurrentUser) -> AuthContext:
        if scope not in ctx.scopes:
            raise HTTPException(
                status_code=403,
                detail={
                    "code": "insufficient_scope",
                    "required": scope,
                    "message": f"scope '{scope}' required",
                },
            )
        return ctx

    return _dep


def assert_tenant(ctx: AuthContext, enterprise_id: int) -> None:
    """在 handler 内调用：校验 path eid 与 token.tenant_id 一致。

    防止跨租户越权：token 是为企业 A 签发的，不能用来读企业 B 的资源。
    """
    if str(enterprise_id) != str(ctx.tenant_id):
        raise HTTPException(
            status_code=403,
            detail={
                "code": "cross_tenant",
                "message": "token tenant does not match path enterprise",
            },
        )
