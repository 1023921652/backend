"""RBAC REST 接口。

prefix=/v1/auth，tags=["auth"]。
所有写操作走事务；AuthError → HTTPException；权限校验走 require_scope 依赖工厂。

测试：
- POST /v1/auth/register → POST /v1/auth/enterprises → POST /v1/auth/login
  → POST /v1/auth/login/enterprise → GET /v1/auth/me
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth import service
from app.auth.deps import (
    AuthContext,
    CurrentUser,
    SessionDep,
    assert_tenant,
    get_current_user,
    require_scope,
)
from app.auth.models import EnterpriseMember, Enterprise, User, UserRole
from app.auth.schemas import (
    BootstrapIn,
    EnterpriseBrief,
    EnterpriseCreate,
    EnterpriseLoginIn,
    EnterpriseOut,
    InviteIn,
    LoginIn,
    LoginOut,
    LogoutIn,
    MeOut,
    MeRole,
    MemberOut,
    MyEnterpriseOut,
    Ok,
    PermissionOut,
    RefreshIn,
    RoleCreate,
    RoleOut,
    RoleUpdate,
    TokenOut,
    UpdateMemberRoleIn,
    UserCreated,
    UserRegister,
)
from app.core.config import settings

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def _raise_from_auth_error(e: service.AuthError) -> None:
    raise HTTPException(
        status_code=e.status,
        detail={"code": e.code, "message": e.message},
    )


# ============ 1. 注册 ============
@router.post("/register", response_model=UserCreated, status_code=status.HTTP_201_CREATED)
async def register_endpoint(body: UserRegister, session: SessionDep):
    try:
        async with session.begin():
            user = await service.register(
                session,
                username=body.username,
                email=body.email,
                password=body.password,
                phone=body.phone,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return UserCreated(user_id=user.id, username=user.username)


# ============ 1b. 一步到位（dev / Swagger 演示用）============
@router.post("/bootstrap", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
async def bootstrap_endpoint(body: BootstrapIn, request: Request, session: SessionDep):
    """注册账号 + 创建企业 + 签发 token，一步完成。

    方便 Swagger 单接口演示，避免 register→create_enterprise→login/enterprise 三步链路。
    生产环境可关闭此接口（通过 ENV=prod 时跳过 router 注册）。
    """
    try:
        async with session.begin():
            user, ent, access_token, refresh_token = await service.bootstrap(
                session,
                username=body.username,
                email=body.email,
                password=body.password,
                phone=body.phone,
                enterprise_name=body.enterprise_name,
                enterprise_slug=body.enterprise_slug,
                device_info=f"bootstrap/{body.enterprise_slug}",
                created_ip=request.client.host if request.client else None,
                created_ua=request.headers.get("user-agent"),
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    from app.auth.security import decode_token
    payload = decode_token(access_token)
    return TokenOut(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
        scopes=list(payload.get("scopes", [])),
        roles=list(payload.get("roles", [])),
    )


# ============ 2. 登录（返回可加入的企业列表）============
@router.post("/login", response_model=LoginOut)
async def login_endpoint(body: LoginIn, session: SessionDep):
    try:
        user, ents = await service.login(
            session, identifier=body.identifier, password=body.password
        )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return LoginOut(
        user_id=user.id,
        username=user.username,
        enterprises=[
            EnterpriseBrief(id=e.id, name=e.name, slug=e.slug, role=r)
            for e, r in ents
        ],
    )


# ============ 3. 企业登录（拿 access + refresh token）============
@router.post("/login/enterprise", response_model=TokenOut)
async def login_enterprise_endpoint(
    body: EnterpriseLoginIn, request: Request, session: SessionDep
):
    try:
        async with session.begin():
            user, ent, access_token, refresh_token = await service.login_enterprise(
                session,
                enterprise_id=body.enterprise_id,
                identifier=body.identifier,
                password=body.password,
                device_info=body.device_info,
                created_ip=request.client.host if request.client else None,
                created_ua=request.headers.get("user-agent"),
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    # collect scopes from token directly (avoid re-query)
    from app.auth.security import decode_token
    payload = decode_token(access_token)
    return TokenOut(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
        scopes=list(payload.get("scopes", [])),
        roles=list(payload.get("roles", [])),
    )


# ============ 4. Refresh token 旋转 ============
@router.post("/refresh", response_model=TokenOut)
async def refresh_endpoint(body: RefreshIn, session: SessionDep):
    try:
        async with session.begin():
            user, ent, access_token, refresh_token = await service.refresh_tokens(
                session, refresh_token=body.refresh_token
            )
    except service.ReplayDetected as e:
        # 重放检测：在独立事务里撤销该用户所有 token（主事务已回滚）
        async with session.begin():
            await service.revoke_all_user_tokens(session, user_id=e.user_id)
        _raise_from_auth_error(e)
    except service.AuthError as e:
        _raise_from_auth_error(e)
    from app.auth.security import decode_token
    payload = decode_token(access_token)
    return TokenOut(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.access_token_expire_minutes * 60,
        scopes=list(payload.get("scopes", [])),
        roles=list(payload.get("roles", [])),
    )


# ============ 5. Logout（撤销 access + refresh token）============
@router.post("/logout", response_model=Ok)
async def logout_endpoint(
    body: LogoutIn,
    session: SessionDep,
    ctx: CurrentUser,
):
    # 撤销当前 access token：写入 Redis 黑名单，TTL=剩余有效期
    from datetime import datetime, timezone
    from app.auth.token_blocklist import revoke_access_jti
    now = int(datetime.now(timezone.utc).timestamp())
    remaining = max(0, ctx.exp - now)
    await revoke_access_jti(ctx.jti, remaining)

    async with session.begin():
        await service.logout(session, refresh_token=body.refresh_token)
    return Ok()


# ============ 6. /me ============
@router.get("/me", response_model=MeOut)
async def me_endpoint(session: SessionDep, ctx: CurrentUser):
    try:
        user, ent, roles, scopes = await service.get_me(
            session, user_id=ctx.user_id, tenant_id=ctx.tenant_id
        )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return MeOut(
        user_id=user.id,
        username=user.username,
        email=user.email,
        enterprise_id=ent.id,
        enterprise_name=ent.name,
        roles=[MeRole(id=r.id, name=r.name, is_builtin=r.is_builtin) for r in roles],
        scopes=scopes,
    )


# ============ 7. 创建企业 ============
@router.post("/enterprises", response_model=EnterpriseOut, status_code=status.HTTP_201_CREATED)
async def create_enterprise_endpoint(
    body: EnterpriseCreate, session: SessionDep, ctx: CurrentUser
):
    try:
        async with session.begin():
            ent = await service.create_enterprise(
                session,
                creator_user_id=ctx.user_id,
                name=body.name,
                slug=body.slug,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return EnterpriseOut.model_validate(ent, from_attributes=True)


# ============ 8. 我加入的企业列表 ============
@router.get("/enterprises", response_model=list[MyEnterpriseOut])
async def list_my_enterprises_endpoint(session: SessionDep, ctx: CurrentUser):
    rows = (
        await session.execute(
            select(Enterprise, Role.name, EnterpriseMember.joined_at)
            .join(EnterpriseMember, EnterpriseMember.enterprise_id == Enterprise.id)
            .outerjoin(UserRole, UserRole.user_id == EnterpriseMember.user_id)
            .outerjoin(
                Role,
                and_(Role.id == UserRole.role_id, Role.enterprise_id == Enterprise.id),
            )
            .where(
                EnterpriseMember.user_id == ctx.user_id,
                EnterpriseMember.status == "active",
            )
            .order_by(EnterpriseMember.joined_at)
        )
    ).all()
    seen: dict[int, MyEnterpriseOut] = {}
    for ent, role_name, joined_at in rows:
        if ent.id in seen:
            continue
        seen[ent.id] = MyEnterpriseOut(
            enterprise=EnterpriseOut.model_validate(ent, from_attributes=True),
            role=role_name,
            joined_at=joined_at,
        )
    return list(seen.values())


# ============ 9. 成员列表 ============
@router.get(
    "/enterprises/{eid}/members",
    response_model=list[MemberOut],
    dependencies=[Depends(require_scope("member:read"))],
)
async def list_members_endpoint(eid: int, session: SessionDep, ctx: CurrentUser):
    assert_tenant(ctx, eid)
    rows = await service.list_members(session, enterprise_id=eid)
    return [
        MemberOut(
            id=em.id,
            user_id=u.id,
            username=u.username,
            email=u.email,
            status=em.status,
            role_name=role_name,
            joined_at=em.joined_at,
        )
        for em, u, role_name in rows
    ]


# ============ 10. 邀请用户加入 ============
@router.post(
    "/enterprises/{eid}/invite",
    response_model=MemberOut,
    status_code=status.HTTP_201_CREATED,
)
async def invite_endpoint(
    eid: int,
    body: InviteIn,
    session: SessionDep,
    ctx: Annotated[AuthContext, Depends(require_scope("member:invite"))],
):
    assert_tenant(ctx, eid)
    try:
        async with session.begin():
            em, user, role = await service.invite(
                session,
                enterprise_id=eid,
                identifier=body.identifier,
                role_name=body.role_name,
                invited_by=ctx.user_id,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return MemberOut(
        id=em.id,
        user_id=user.id,
        username=user.username,
        email=user.email,
        status=em.status,
        role_name=role.name,
        joined_at=em.joined_at,
    )


# ============ 11. 改成员角色 ============
@router.put("/enterprises/{eid}/members/{uid}/role", response_model=MemberOut)
async def update_member_role_endpoint(
    eid: int,
    uid: int,
    body: UpdateMemberRoleIn,
    session: SessionDep,
    ctx: Annotated[AuthContext, Depends(require_scope("member:update"))],
):
    assert_tenant(ctx, eid)
    try:
        async with session.begin():
            em, role = await service.update_member_role(
                session,
                enterprise_id=eid,
                user_id=uid,
                role_name=body.role_name,
                actor_id=ctx.user_id,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    # 查一下用户名（service 没返回）
    u = await session.get(User, uid)
    return MemberOut(
        id=em.id,
        user_id=uid,
        username=u.username if u else "",
        email=u.email if u else "",
        status=em.status,
        role_name=role.name,
        joined_at=em.joined_at,
    )


# ============ 12. 踢出成员 ============
@router.delete("/enterprises/{eid}/members/{uid}", response_model=Ok)
async def remove_member_endpoint(
    eid: int,
    uid: int,
    session: SessionDep,
    ctx: Annotated[AuthContext, Depends(require_scope("member:remove"))],
):
    assert_tenant(ctx, eid)
    try:
        async with session.begin():
            await service.remove_member(
                session, enterprise_id=eid, user_id=uid, actor_id=ctx.user_id
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return Ok()


# ============ 13. 角色列表 ============
@router.get(
    "/enterprises/{eid}/roles",
    response_model=list[RoleOut],
    dependencies=[Depends(require_scope("role:read"))],
)
async def list_roles_endpoint(eid: int, session: SessionDep, ctx: CurrentUser):
    assert_tenant(ctx, eid)
    rows = await service.list_roles(session, enterprise_id=eid)
    return [
        RoleOut(
            id=role.id,
            name=role.name,
            description=role.description,
            is_builtin=role.is_builtin,
            scopes=scopes,
        )
        for role, scopes in rows
    ]


# ============ 14. 新建角色 ============
@router.post(
    "/enterprises/{eid}/roles",
    response_model=RoleOut,
    status_code=status.HTTP_201_CREATED,
)
async def create_role_endpoint(
    eid: int,
    body: RoleCreate,
    session: SessionDep,
    ctx: Annotated[AuthContext, Depends(require_scope("role:create"))],
):
    assert_tenant(ctx, eid)
    try:
        async with session.begin():
            role, scopes = await service.create_role(
                session,
                enterprise_id=eid,
                name=body.name,
                description=body.description,
                scopes=body.scopes,
                created_by=ctx.user_id,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return RoleOut(
        id=role.id,
        name=role.name,
        description=role.description,
        is_builtin=role.is_builtin,
        scopes=scopes,
    )


# ============ 15. 改角色 scopes ============
@router.put("/enterprises/{eid}/roles/{rid}", response_model=RoleOut)
async def update_role_endpoint(
    eid: int,
    rid: int,
    body: RoleUpdate,
    session: SessionDep,
    ctx: Annotated[AuthContext, Depends(require_scope("role:update"))],
):
    assert_tenant(ctx, eid)
    try:
        async with session.begin():
            role, scopes = await service.update_role(
                session,
                enterprise_id=eid,
                role_id=rid,
                description=body.description,
                scopes=body.scopes,
            )
    except service.AuthError as e:
        _raise_from_auth_error(e)
    return RoleOut(
        id=role.id,
        name=role.name,
        description=role.description,
        is_builtin=role.is_builtin,
        scopes=scopes,
    )


# ============ 16. 全局权限字典 ============
@router.get(
    "/permissions",
    response_model=list[PermissionOut],
    dependencies=[Depends(require_scope("permission:read"))],
)
async def list_permissions_endpoint(session: SessionDep):
    return [
        PermissionOut.model_validate(p, from_attributes=True)
        for p in await service.list_permissions(session)
    ]


# ============ 17. Demo：演示 require_scope ============
@router.get(
    "/demo/check-scope",
    dependencies=[Depends(require_scope("user:read"))],
)
async def demo_check_scope_endpoint(
    ctx: CurrentUser,
    scope: Annotated[Optional[str], Query()] = None,
):
    """演示依赖工厂效果：路由声明 dependencies=[require_scope('user:read')]，
    无该 scope 的 token 直接被 403 拦截。
    """
    return {
        "has_scope": scope in ctx.scopes if scope else None,
        "current_scopes": ctx.scopes,
        "queried_scope": scope,
        "user_id": ctx.user_id,
        "tenant_id": ctx.tenant_id,
    }
