"""RBAC Pydantic v2 入参/出参 DTO。"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ============ 通用 ============
class Ok(BaseModel):
    ok: bool = True


# ============ Auth ============
class UserRegister(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    phone: Optional[str] = Field(default=None, max_length=20)

    @field_validator("password")
    @classmethod
    def _check_pwd(cls, v: str) -> str:
        from app.auth.security import validate_password_strength
        try:
            validate_password_strength(v)
        except ValueError as e:
            raise ValueError(str(e)) from e
        return v


class UserCreated(BaseModel):
    user_id: int
    username: str


class LoginIn(BaseModel):
    """identifier 可以是 username / email / phone。"""
    identifier: str = Field(min_length=1)
    password: str


class EnterpriseBrief(BaseModel):
    id: int
    name: str
    slug: str
    role: Optional[str] = None  # 该用户在此企业的角色名（取首个）


class LoginOut(BaseModel):
    user_id: int
    username: str
    enterprises: list[EnterpriseBrief]


class EnterpriseLoginIn(BaseModel):
    enterprise_id: int
    identifier: str
    password: str
    device_info: Optional[str] = None


class BootstrapIn(BaseModel):
    """一步到位：注册账号 + 创建企业 + 签发 token。仅限 dev / 首次部署用。"""
    username: str = Field(min_length=3, max_length=64)
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    phone: Optional[str] = Field(default=None, max_length=20)
    enterprise_name: str = Field(min_length=1, max_length=128)
    enterprise_slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")

    @field_validator("password")
    @classmethod
    def _check_pwd(cls, v: str) -> str:
        from app.auth.security import validate_password_strength
        try:
            validate_password_strength(v)
        except ValueError as e:
            raise ValueError(str(e)) from e
        return v


class TokenOut(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int  # 秒
    scopes: list[str]
    roles: list[str]


class RefreshIn(BaseModel):
    refresh_token: str


class LogoutIn(BaseModel):
    refresh_token: Optional[str] = None


class MeRole(BaseModel):
    id: int
    name: str
    is_builtin: bool


class MeOut(BaseModel):
    user_id: int
    username: str
    email: str
    enterprise_id: int
    enterprise_name: str
    roles: list[MeRole]
    scopes: list[str]


# ============ Enterprise ============
class EnterpriseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    slug: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")


class EnterpriseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    slug: str
    owner_user_id: Optional[int]
    status: str


class MyEnterpriseOut(BaseModel):
    enterprise: EnterpriseOut
    role: Optional[str]
    joined_at: datetime


# ============ Member ============
class MemberOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    username: str
    email: str
    status: str
    role_name: Optional[str] = None
    joined_at: datetime


class InviteIn(BaseModel):
    identifier: str  # username / email / phone
    role_name: str
    message: Optional[str] = None


class UpdateMemberRoleIn(BaseModel):
    role_name: str


# ============ Role ============
class RoleOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    description: Optional[str]
    is_builtin: bool
    scopes: list[str] = []


class RoleCreate(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    description: Optional[str] = Field(default=None, max_length=255)
    scopes: list[str] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    description: Optional[str] = None
    scopes: Optional[list[str]] = None


# ============ Permission ============
class PermissionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    scope: str
    resource: str
    action: str
    description: Optional[str] = None
