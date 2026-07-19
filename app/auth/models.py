"""多租户 RBAC ORM 模型。

8 张表：
- users              全局账号（类似阿里云/腾讯云账号）
- enterprises        租户
- enterprise_members 用户 × 企业（join table，记录在该企业的状态）
- roles              角色（per-enterprise，含 is_builtin 的 owner/admin/member）
- permissions        全局权限字典（scope = resource:action）
- role_permissions   角色 × 权限（M2M）
- user_roles         用户 × 角色（通过 role.enterprise_id 间接隔离租户）
- refresh_tokens     JWT refresh token 持久化（jti + token_hash + revoked）

约定：
- 主键统一 BIGINT 自增
- 所有表带 created_at / updated_at / deleted_at（软删除）
- 时间戳在应用层用 datetime.now(timezone.utc)，DB 端 server_default=now() 兜底
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.mysql.base import Base


# ============ users ============
class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("username", name="uq_users_username"),
        UniqueConstraint("email", name="uq_users_email"),
        UniqueConstraint("phone", name="uq_users_phone"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False)
    email: Mapped[str] = mapped_column(String(128), nullable=False)
    phone: Mapped[Optional[str]] = mapped_column(String(20))
    pwd_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # active / disabled
    last_login_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    memberships: Mapped[list[EnterpriseMember]] = relationship(
        back_populates="user",
        foreign_keys="EnterpriseMember.user_id",
        cascade="all, delete-orphan",
    )


# ============ enterprises ============
class Enterprise(Base):
    __tablename__ = "enterprises"
    __table_args__ = (
        UniqueConstraint("slug", name="uq_enterprises_slug"),
        Index("ix_enterprises_owner", "owner_user_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    slug: Mapped[str] = mapped_column(String(64), nullable=False)
    owner_user_id: Mapped[Optional[int]] = mapped_column(
        BigInteger, ForeignKey("users.id"), nullable=True
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # active / frozen
    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    members: Mapped[list[EnterpriseMember]] = relationship(back_populates="enterprise")
    roles: Mapped[list[Role]] = relationship(back_populates="enterprise")


# ============ enterprise_members ============
class EnterpriseMember(Base):
    """用户在某企业的成员关系。

    user_roles 通过 role.enterprise_id 间接隔离租户；本表只承载成员状态/邀请。
    """
    __tablename__ = "enterprise_members"
    __table_args__ = (
        UniqueConstraint("user_id", "enterprise_id", name="uq_em_user_enterprise"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    enterprise_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enterprises.id"), nullable=False
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")  # active / invited / left
    invited_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    joined_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    user: Mapped[User] = relationship(back_populates="memberships", foreign_keys=[user_id])
    enterprise: Mapped[Enterprise] = relationship(back_populates="members")


# ============ roles ============
class Role(Base):
    """角色：per-enterprise。

    is_builtin=True 的 owner / admin / member 在企业创建时自动生成，禁止删除/改名/改 scopes。
    """
    __tablename__ = "roles"
    __table_args__ = (
        UniqueConstraint("enterprise_id", "name", name="uq_roles_enterprise_name"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    enterprise_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enterprises.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    is_builtin: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)

    enterprise: Mapped[Enterprise] = relationship(back_populates="roles")
    permissions: Mapped[list[RolePermission]] = relationship(
        back_populates="role", cascade="all, delete-orphan"
    )


# ============ permissions ============
class Permission(Base):
    """全局权限字典表。

    scope 字符串格式 `resource:action`，全局共享（不绑定 enterprise）。
    首次启动 service.upsert_permissions 从 app.auth.permissions.PERMISSIONS upsert。
    """
    __tablename__ = "permissions"
    __table_args__ = (
        UniqueConstraint("scope", name="uq_permissions_scope"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    resource: Mapped[str] = mapped_column(String(32), nullable=False)
    action: Mapped[str] = mapped_column(String(32), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


# ============ role_permissions ============
class RolePermission(Base):
    """角色 × 权限（M2M）。"""
    __tablename__ = "role_permissions"

    role_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("roles.id"), primary_key=True
    )
    permission_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("permissions.id"), primary_key=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    role: Mapped[Role] = relationship(back_populates="permissions")
    permission: Mapped[Permission] = relationship()


# ============ user_roles ============
class UserRole(Base):
    """用户在某企业内的角色授权。

    通过 role.enterprise_id 间接隔离租户：同一 user_id 在不同企业可有不同 role_id。
    """
    __tablename__ = "user_roles"
    __table_args__ = (
        UniqueConstraint("user_id", "role_id", name="uq_ur_user_role"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    role_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("roles.id"), nullable=False)
    granted_by: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("users.id"))
    granted_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime)


# ============ refresh_tokens ============
class RefreshToken(Base):
    """Refresh token 持久化记录。

    - jti：JWT ID，唯一
    - token_hash：SHA256(token 原值)，防止原值泄漏（DB 落盘的是哈希）
    - revoked_at：撤销时间戳；非空表示已撤销（黑名单）
    - expires_at：过期时间
    """
    __tablename__ = "refresh_tokens"
    __table_args__ = (
        Index("ix_rt_user", "user_id"),
        Index("ix_rt_jti", "jti"),
        Index("ix_rt_expires", "expires_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    jti: Mapped[str] = mapped_column(String(36), nullable=False)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=False)
    enterprise_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("enterprises.id"), nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    device_info: Mapped[Optional[str]] = mapped_column(String(255))
    created_ip: Mapped[Optional[str]] = mapped_column(String(45))
    created_ua: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


__all__ = [
    "User",
    "Enterprise",
    "EnterpriseMember",
    "Role",
    "Permission",
    "RolePermission",
    "UserRole",
    "RefreshToken",
]
