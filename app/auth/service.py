"""RBAC 业务逻辑层。

所有写操作走 `async with session.begin():`；多步操作用事务保证原子。
关键流程：
- create_enterprise：企业创建 → 自动产出 owner/admin/member 角色 → 把创建者设为 owner
- update_member_role / remove_member：含 owner 唯一性校验（每企业至少一名 owner）
- refresh：token 旋转，撤销旧 jti 后签发新对
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import and_, false as sql_false, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.auth.models import (
    Enterprise,
    EnterpriseMember,
    Permission,
    RefreshToken,
    Role,
    RolePermission,
    User,
    UserRole,
)
from app.auth.permissions import PERMISSIONS, ROLE_TEMPLATES, BUILTIN_ROLE_NAMES
from app.auth.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    hash_token,
    verify_password,
)
from app.core.config import settings


def _utcnow_naive() -> datetime:
    """DB 列无 tz，统一返回 naive UTC 时间避免比较错误。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ============ 异常类型（api 层转 HTTPException）============
class AuthError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code = code
        self.message = message
        self.status = status


# ============ Permissions 字典 bootstrap ============
async def upsert_permissions(session: AsyncSession) -> None:
    """从 PERMISSIONS 字典 upsert 到 permissions 表。

    幂等：已存在的 scope 跳过。
    """
    existing = (
        await session.execute(select(Permission.scope))
    ).scalars().all()
    existing_set = set(existing)
    for scope, (resource, action) in PERMISSIONS.items():
        if scope in existing_set:
            continue
        session.add(
            Permission(scope=scope, resource=resource, action=action, description=scope)
        )


async def _get_permission_ids(session: AsyncSession, scopes: set[str]) -> dict[str, int]:
    rows = (
        await session.execute(
            select(Permission.id, Permission.scope).where(Permission.scope.in_(scopes))
        )
    ).all()
    return {scope: pid for pid, scope in rows}


# ============ 注册 ============
async def register(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    phone: Optional[str] = None,
) -> User:
    # 唯一性预检（DB uq 兜底，但提前报 409 更友好）
    phone_cond = (User.phone == phone) if phone else sql_false()
    conflict = (
        await session.execute(
            select(User).where(
                (User.username == username) | (User.email == email) | phone_cond
            )
        )
    ).scalar_one_or_none()
    if conflict:
        if conflict.username == username:
            raise AuthError("username_taken", "username already exists", 409)
        if conflict.email == email:
            raise AuthError("email_taken", "email already exists", 409)
        raise AuthError("phone_taken", "phone already exists", 409)

    user = User(
        username=username,
        email=email,
        phone=phone,
        pwd_hash=hash_password(password),
        status="active",
    )
    session.add(user)
    await session.flush()
    return user


# ============ 登录 ============
async def _find_user_by_identifier(session: AsyncSession, identifier: str) -> Optional[User]:
    return (
        await session.execute(
            select(User).where(
                (User.username == identifier)
                | (User.email == identifier)
                | (User.phone == identifier)
            )
        )
    ).scalar_one_or_none()


async def login(
    session: AsyncSession, *, identifier: str, password: str
) -> tuple[User, list[tuple[Enterprise, Optional[str]]]]:
    """返回用户 + 该用户加入的所有企业（含每个企业的首个角色名）。"""
    user = await _find_user_by_identifier(session, identifier)
    if not user or not verify_password(password, user.pwd_hash):
        raise AuthError("invalid_credentials", "invalid username or password", 401)
    if user.status != "active":
        raise AuthError("user_disabled", "user is disabled", 403)

    rows = (
        await session.execute(
            select(Enterprise, Role.name)
            .join(EnterpriseMember, EnterpriseMember.enterprise_id == Enterprise.id)
            .outerjoin(UserRole, UserRole.user_id == EnterpriseMember.user_id)
            .outerjoin(Role, and_(Role.id == UserRole.role_id, Role.enterprise_id == Enterprise.id))
            .where(EnterpriseMember.user_id == user.id, EnterpriseMember.status == "active")
            .order_by(Enterprise.id)
        )
    ).all()
    # 同一企业可能出现多行（多角色），聚合取首个
    by_eid: dict[int, tuple[Enterprise, Optional[str]]] = {}
    for ent, role_name in rows:
        if ent.id not in by_eid:
            by_eid[ent.id] = (ent, role_name)
    return user, list(by_eid.values())


async def collect_user_roles_scopes(
    session: AsyncSession, *, user_id: int, enterprise_id: int
) -> tuple[list[Role], list[str]]:
    """聚合用户在某企业的所有角色 + 展开后的 scope 集合。"""
    rows = (
        await session.execute(
            select(Role, Permission.scope)
            .join(UserRole, UserRole.role_id == Role.id)
            .join(RolePermission, RolePermission.role_id == Role.id, isouter=True)
            .join(Permission, Permission.id == RolePermission.permission_id, isouter=True)
            .where(UserRole.user_id == user_id, Role.enterprise_id == enterprise_id)
            .order_by(Role.id)
        )
    ).all()
    roles_by_id: dict[int, Role] = {}
    scopes: set[str] = set()
    for role, scope in rows:
        roles_by_id.setdefault(role.id, role)
        if scope:
            scopes.add(scope)
    return list(roles_by_id.values()), sorted(scopes)


async def login_enterprise(
    session: AsyncSession,
    *,
    enterprise_id: int,
    identifier: str,
    password: str,
    device_info: Optional[str] = None,
    created_ip: Optional[str] = None,
    created_ua: Optional[str] = None,
) -> tuple[User, Enterprise, str, str]:
    """企业登录：返回 (user, enterprise, access_token, refresh_token)。

    校验：用户存在 + 密码正确 + 用户是该企业成员 + 企业状态正常。
    """
    user = await _find_user_by_identifier(session, identifier)
    if not user or not verify_password(password, user.pwd_hash):
        raise AuthError("invalid_credentials", "invalid username or password", 401)
    if user.status != "active":
        raise AuthError("user_disabled", "user is disabled", 403)

    ent = (
        await session.execute(
            select(Enterprise).where(Enterprise.id == enterprise_id)
        )
    ).scalar_one_or_none()
    if not ent or ent.deleted_at is not None:
        raise AuthError("enterprise_not_found", "enterprise not found", 404)
    if ent.status != "active":
        raise AuthError("enterprise_frozen", "enterprise is frozen", 403)

    membership = (
        await session.execute(
            select(EnterpriseMember).where(
                EnterpriseMember.user_id == user.id,
                EnterpriseMember.enterprise_id == enterprise_id,
                EnterpriseMember.status == "active",
            )
        )
    ).scalar_one_or_none()
    if not membership:
        raise AuthError("not_member", "user is not a member of this enterprise", 403)

    roles, scopes = await collect_user_roles_scopes(
        session, user_id=user.id, enterprise_id=enterprise_id
    )
    return await _issue_tokens(
        session,
        user=user,
        enterprise=ent,
        roles=roles,
        scopes=scopes,
        device_info=device_info,
        created_ip=created_ip,
        created_ua=created_ua,
    )


async def _issue_tokens(
    session: AsyncSession,
    *,
    user: User,
    enterprise: Enterprise,
    roles: list[Role],
    scopes: list[str],
    device_info: Optional[str] = None,
    created_ip: Optional[str] = None,
    created_ua: Optional[str] = None,
) -> tuple[User, Enterprise, str, str]:
    """统一签发 access + refresh token 并落盘 refresh 记录。"""
    role_names = [r.name for r in roles]
    access_token, _, _ = create_access_token(
        sub=user.id, tenant_id=enterprise.id, roles=role_names, scopes=scopes
    )
    refresh_token, jti, expires_at = create_refresh_token(
        sub=user.id, tenant_id=enterprise.id
    )
    session.add(
        RefreshToken(
            jti=jti,
            token_hash=hash_token(refresh_token),
            user_id=user.id,
            enterprise_id=enterprise.id,
            expires_at=expires_at,
            device_info=device_info,
            created_ip=created_ip,
            created_ua=created_ua,
        )
    )
    await session.execute(
        update(User).where(User.id == user.id).values(last_login_at=_utcnow_naive())
    )
    return user, enterprise, access_token, refresh_token


async def bootstrap(
    session: AsyncSession,
    *,
    username: str,
    email: str,
    password: str,
    phone: Optional[str],
    enterprise_name: str,
    enterprise_slug: str,
    device_info: Optional[str] = None,
    created_ip: Optional[str] = None,
    created_ua: Optional[str] = None,
) -> tuple[User, Enterprise, str, str]:
    """一步到位：注册全局账号 + 创建企业（自动 owner 角色 + 把创建者设为 owner）+ 签发 token。

    全流程在同一个事务内（由 api 层 `async with session.begin()` 包裹），任一步失败回滚。
    用于首次部署 / Swagger 演示，避免 register→create_enterprise→login_enterprise 三步链路的鸡生蛋问题。
    """
    user = await register(
        session,
        username=username,
        email=email,
        password=password,
        phone=phone,
    )
    ent = await create_enterprise(
        session,
        creator_user_id=user.id,
        name=enterprise_name,
        slug=enterprise_slug,
    )
    await session.flush()  # autoflush=False，确保 user_roles / role_permissions 落盘后再查
    roles, scopes = await collect_user_roles_scopes(
        session, user_id=user.id, enterprise_id=ent.id
    )
    return await _issue_tokens(
        session,
        user=user,
        enterprise=ent,
        roles=roles,
        scopes=scopes,
        device_info=device_info,
        created_ip=created_ip,
        created_ua=created_ua,
    )


# ============ Token 旋转 ============
async def refresh_tokens(
    session: AsyncSession, *, refresh_token: str
) -> tuple[User, Enterprise, str, str]:
    """refresh token 旋转：撤销旧 jti + 签发新对。"""
    try:
        payload = decode_token(refresh_token)
    except Exception as e:
        raise AuthError("invalid_token", f"invalid refresh token: {e}", 401)

    if payload.get("type") != "refresh":
        raise AuthError("wrong_token_type", "not a refresh token", 401)

    jti = payload.get("jti")
    user_id = int(payload.get("sub"))
    tenant_id = int(payload.get("tenant_id"))
    token_hash = hash_token(refresh_token)

    rec = (
        await session.execute(
            select(RefreshToken).where(RefreshToken.jti == jti)
        )
    ).scalar_one_or_none()
    if not rec or rec.token_hash != token_hash:
        raise AuthError("token_not_found", "refresh token not recognized", 401)
    if rec.revoked_at is not None:
        # 重放：可能是泄漏，撤销整个用户的所有 token
        await session.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=_utcnow_naive())
        )
        raise AuthError("token_replayed", "refresh token replay detected; all tokens revoked", 401)
    if rec.expires_at < _utcnow_naive():
        raise AuthError("token_expired", "refresh token expired", 401)

    # 撤销旧 + 签发新
    rec.revoked_at = _utcnow_naive()

    user = await session.get(User, user_id)
    if not user or user.status != "active":
        raise AuthError("user_unavailable", "user no longer active", 401)

    ent = await session.get(Enterprise, tenant_id)
    if not ent or ent.deleted_at is not None or ent.status != "active":
        raise AuthError("enterprise_unavailable", "enterprise no longer active", 401)

    roles, scopes = await collect_user_roles_scopes(
        session, user_id=user_id, enterprise_id=tenant_id
    )
    access_token, _, _ = create_access_token(
        sub=user_id, tenant_id=tenant_id, roles=[r.name for r in roles], scopes=scopes
    )
    new_refresh, new_jti, expires_at = create_refresh_token(
        sub=user_id, tenant_id=tenant_id
    )
    session.add(
        RefreshToken(
            jti=new_jti,
            token_hash=hash_token(new_refresh),
            user_id=user_id,
            enterprise_id=tenant_id,
            expires_at=expires_at,
        )
    )
    return user, ent, access_token, new_refresh


async def logout(session: AsyncSession, *, refresh_token: Optional[str]) -> None:
    """撤销 refresh token（如有）。"""
    if not refresh_token:
        return
    try:
        payload = decode_token(refresh_token)
    except Exception:
        return
    if payload.get("type") != "refresh":
        return
    jti = payload.get("jti")
    await session.execute(
        update(RefreshToken)
        .where(RefreshToken.jti == jti, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=_utcnow_naive())
    )


# ============ 创建企业（原子流程）============
async def create_enterprise(
    session: AsyncSession, *, creator_user_id: int, name: str, slug: str
) -> Enterprise:
    """7 步原子流程：
    1. upsert permissions 字典
    2. INSERT enterprises（owner_user_id 暂空）
    3. INSERT roles（owner / admin / member，is_builtin=True）
    4. INSERT role_permissions（按 ROLE_TEMPLATES）
    5. INSERT enterprise_members（creator，status=active）
    6. INSERT user_roles（creator → owner）
    7. UPDATE enterprises.owner_user_id
    """
    # slug 唯一性预检
    exist = (
        await session.execute(select(Enterprise).where(Enterprise.slug == slug))
    ).scalar_one_or_none()
    if exist:
        raise AuthError("slug_taken", "slug already exists", 409)

    await upsert_permissions(session)

    ent = Enterprise(name=name, slug=slug, owner_user_id=None, created_by=creator_user_id)
    session.add(ent)
    await session.flush()  # 拿到 ent.id

    perm_id_map = await _get_permission_ids(
        session, set().union(*[set(s) for s in ROLE_TEMPLATES.values()])
    )

    role_by_name: dict[str, Role] = {}
    for role_name, scopes in ROLE_TEMPLATES.items():
        role = Role(
            enterprise_id=ent.id,
            name=role_name,
            description=f"built-in {role_name} role",
            is_builtin=True,
            created_by=creator_user_id,
        )
        session.add(role)
        await session.flush()
        role_by_name[role_name] = role
        for scope in scopes:
            pid = perm_id_map.get(scope)
            if pid is not None:
                session.add(RolePermission(role_id=role.id, permission_id=pid))

    # creator 加入企业
    membership = EnterpriseMember(
        user_id=creator_user_id,
        enterprise_id=ent.id,
        status="active",
        invited_by=creator_user_id,
    )
    session.add(membership)

    # creator 关联 owner 角色
    owner_role = role_by_name["owner"]
    session.add(
        UserRole(
            user_id=creator_user_id,
            role_id=owner_role.id,
            granted_by=creator_user_id,
        )
    )

    # 回填 owner_user_id
    ent.owner_user_id = creator_user_id
    return ent


# ============ 成员管理 ============
async def list_members(session: AsyncSession, *, enterprise_id: int) -> list[tuple[EnterpriseMember, User, Optional[str]]]:
    rows = (
        await session.execute(
            select(EnterpriseMember, User, Role.name)
            .join(User, User.id == EnterpriseMember.user_id)
            .outerjoin(UserRole, UserRole.user_id == EnterpriseMember.user_id)
            .outerjoin(
                Role,
                and_(Role.id == UserRole.role_id, Role.enterprise_id == enterprise_id),
            )
            .where(EnterpriseMember.enterprise_id == enterprise_id)
            .order_by(EnterpriseMember.joined_at)
        )
    ).all()
    # 同一 user 可能因多角色出现多行，仅保留首个（角色名）
    seen: set[int] = set()
    out = []
    for em, u, role_name in rows:
        if em.user_id in seen:
            continue
        seen.add(em.user_id)
        out.append((em, u, role_name))
    return out


async def _find_role_by_name(
    session: AsyncSession, *, enterprise_id: int, role_name: str
) -> Role:
    role = (
        await session.execute(
            select(Role).where(
                Role.enterprise_id == enterprise_id, Role.name == role_name
            )
        )
    ).scalar_one_or_none()
    if not role:
        raise AuthError("role_not_found", f"role '{role_name}' not found", 404)
    return role


async def _find_user_by_identifier_eager(
    session: AsyncSession, identifier: str
) -> User:
    user = await _find_user_by_identifier(session, identifier)
    if not user:
        raise AuthError("user_not_found", "user not found", 404)
    return user


async def invite(
    session: AsyncSession,
    *,
    enterprise_id: int,
    identifier: str,
    role_name: str,
    invited_by: int,
) -> tuple[EnterpriseMember, User, Role]:
    user = await _find_user_by_identifier_eager(session, identifier)
    role = await _find_role_by_name(
        session, enterprise_id=enterprise_id, role_name=role_name
    )

    existing = (
        await session.execute(
            select(EnterpriseMember).where(
                EnterpriseMember.user_id == user.id,
                EnterpriseMember.enterprise_id == enterprise_id,
            )
        )
    ).scalar_one_or_none()
    if existing:
        if existing.status == "active":
            raise AuthError("already_member", "user is already a member", 409)
        # 之前离开过，重新激活
        existing.status = "active"
        em = existing
    else:
        em = EnterpriseMember(
            user_id=user.id,
            enterprise_id=enterprise_id,
            status="active",
            invited_by=invited_by,
        )
        session.add(em)
        await session.flush()
        await session.refresh(em)  # 拉回 server_default 的 joined_at

    # 删除该用户在此企业的旧角色授权，绑定新角色
    await session.execute(
        UserRole.__table__.delete().where(
            UserRole.user_id == user.id,
            UserRole.role_id.in_(
                select(Role.id).where(Role.enterprise_id == enterprise_id)
            ),
        )
    )
    await session.flush()
    session.add(UserRole(user_id=user.id, role_id=role.id, granted_by=invited_by))
    return em, user, role


async def count_active_owners(
    session: AsyncSession, *, enterprise_id: int
) -> int:
    """统计该企业当前 active 的 owner 数。"""
    cnt = (
        await session.execute(
            select(func.count(UserRole.id))
            .join(Role, Role.id == UserRole.role_id)
            .join(EnterpriseMember, and_(
                EnterpriseMember.user_id == UserRole.user_id,
                EnterpriseMember.enterprise_id == enterprise_id,
                EnterpriseMember.status == "active",
            ))
            .where(Role.enterprise_id == enterprise_id, Role.name == "owner")
        )
    ).scalar_one()
    return int(cnt or 0)


async def update_member_role(
    session: AsyncSession,
    *,
    enterprise_id: int,
    user_id: int,
    role_name: str,
    actor_id: int,
) -> tuple[EnterpriseMember, Role]:
    em = (
        await session.execute(
            select(EnterpriseMember).where(
                EnterpriseMember.user_id == user_id,
                EnterpriseMember.enterprise_id == enterprise_id,
                EnterpriseMember.status == "active",
            )
        )
    ).scalar_one_or_none()
    if not em:
        raise AuthError("member_not_found", "member not found", 404)

    role = await _find_role_by_name(
        session, enterprise_id=enterprise_id, role_name=role_name
    )

    # owner 唯一性校验：如果当前用户是 owner 且新角色不是 owner，且他是唯一 owner → 拒绝
    current_owner_role = (
        await session.execute(
            select(UserRole).join(Role, Role.id == UserRole.role_id).where(
                UserRole.user_id == user_id,
                Role.enterprise_id == enterprise_id,
                Role.name == "owner",
            )
        )
    ).scalar_one_or_none()
    if current_owner_role and role.name != "owner":
        owner_count = await count_active_owners(session, enterprise_id=enterprise_id)
        if owner_count <= 1:
            raise AuthError(
                "last_owner",
                "cannot demote the last owner; assign another owner first",
                409,
            )

    # 删旧角色 + 加新
    await session.execute(
        UserRole.__table__.delete().where(
            UserRole.user_id == user_id,
            UserRole.role_id.in_(
                select(Role.id).where(Role.enterprise_id == enterprise_id)
            ),
        )
    )
    session.add(UserRole(user_id=user_id, role_id=role.id, granted_by=actor_id))
    return em, role


async def remove_member(
    session: AsyncSession, *, enterprise_id: int, user_id: int, actor_id: int
) -> None:
    em = (
        await session.execute(
            select(EnterpriseMember).where(
                EnterpriseMember.user_id == user_id,
                EnterpriseMember.enterprise_id == enterprise_id,
                EnterpriseMember.status == "active",
            )
        )
    ).scalar_one_or_none()
    if not em:
        raise AuthError("member_not_found", "member not found", 404)

    # owner 唯一性
    is_owner = (
        await session.execute(
            select(UserRole).join(Role, Role.id == UserRole.role_id).where(
                UserRole.user_id == user_id,
                Role.enterprise_id == enterprise_id,
                Role.name == "owner",
            )
        )
    ).scalar_one_or_none()
    if is_owner:
        owner_count = await count_active_owners(session, enterprise_id=enterprise_id)
        if owner_count <= 1:
            raise AuthError(
                "last_owner",
                "cannot remove the last owner; transfer ownership first",
                409,
            )

    em.status = "left"
    em.updated_at = _utcnow_naive()
    # 清理该用户在此企业的角色授权
    await session.execute(
        UserRole.__table__.delete().where(
            UserRole.user_id == user_id,
            UserRole.role_id.in_(
                select(Role.id).where(Role.enterprise_id == enterprise_id)
            ),
        )
    )


# ============ 角色管理 ============
async def list_roles(session: AsyncSession, *, enterprise_id: int) -> list[tuple[Role, list[str]]]:
    """返回企业内所有角色 + 每个角色的 scope 列表。"""
    roles = (
        await session.execute(
            select(Role)
            .options(selectinload(Role.permissions).selectinload(RolePermission.permission))
            .where(Role.enterprise_id == enterprise_id)
            .order_by(Role.id)
        )
    ).scalars().all()
    return [(r, [rp.permission.scope for rp in r.permissions]) for r in roles]


async def create_role(
    session: AsyncSession,
    *,
    enterprise_id: int,
    name: str,
    description: Optional[str],
    scopes: list[str],
    created_by: int,
) -> tuple[Role, list[str]]:
    if name in BUILTIN_ROLE_NAMES:
        raise AuthError("reserved_role_name", f"'{name}' is reserved for builtin roles", 400)

    existing = (
        await session.execute(
            select(Role).where(
                Role.enterprise_id == enterprise_id, Role.name == name
            )
        )
    ).scalar_one_or_none()
    if existing:
        raise AuthError("role_taken", "role name already exists in this enterprise", 409)

    await upsert_permissions(session)
    role = Role(
        enterprise_id=enterprise_id,
        name=name,
        description=description,
        is_builtin=False,
        created_by=created_by,
    )
    session.add(role)
    await session.flush()

    perm_ids = await _get_permission_ids(session, set(scopes))
    for scope, pid in perm_ids.items():
        session.add(RolePermission(role_id=role.id, permission_id=pid))
    return role, sorted(perm_ids.keys())


async def update_role(
    session: AsyncSession,
    *,
    enterprise_id: int,
    role_id: int,
    description: Optional[str] = None,
    scopes: Optional[list[str]] = None,
) -> tuple[Role, list[str]]:
    role = (
        await session.execute(
            select(Role)
            .options(selectinload(Role.permissions).selectinload(RolePermission.permission))
            .where(Role.enterprise_id == enterprise_id, Role.id == role_id)
        )
    ).scalar_one_or_none()
    if not role:
        raise AuthError("role_not_found", "role not found", 404)
    if role.is_builtin:
        raise AuthError("builtin_role_locked", "builtin roles cannot be modified", 400)

    if description is not None:
        role.description = description

    final_scopes: Optional[list[str]] = None
    if scopes is not None:
        await upsert_permissions(session)
        # 删旧关联 + 加新
        await session.execute(
            RolePermission.__table__.delete().where(RolePermission.role_id == role.id)
        )
        await session.flush()
        perm_ids = await _get_permission_ids(session, set(scopes))
        for scope, pid in perm_ids.items():
            session.add(RolePermission(role_id=role.id, permission_id=pid))
        final_scopes = sorted(perm_ids.keys())

    await session.flush()
    return role, final_scopes if final_scopes is not None else [rp.permission.scope for rp in role.permissions]


# ============ Permissions 字典查询 ============
async def list_permissions(session: AsyncSession) -> list[Permission]:
    return list(
        (
            await session.execute(
                select(Permission).order_by(Permission.resource, Permission.action)
            )
        ).scalars().all()
    )


# ============ Me ============
async def get_me(
    session: AsyncSession, *, user_id: int, tenant_id: int
) -> tuple[User, Enterprise, list[Role], list[str]]:
    user = await session.get(User, user_id)
    if not user:
        raise AuthError("user_not_found", "user not found", 404)
    ent = await session.get(Enterprise, tenant_id)
    if not ent:
        raise AuthError("enterprise_not_found", "enterprise not found", 404)
    roles, scopes = await collect_user_roles_scopes(
        session, user_id=user_id, enterprise_id=tenant_id
    )
    return user, ent, roles, scopes
