"""密码哈希 + JWT 签发/校验。

密码：pwdlib Argon2（已装依赖，不要换 passlib/bcrypt）。
JWT：pyjwt（HS256 默认；如需 RS256 改 settings.jwt_algorithm + 私钥来源）。
"""
from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher

from app.core.config import settings

# ============ 密码哈希 ============
_pwd_hasher = PasswordHash((Argon2Hasher(),))


def hash_password(plain: str) -> str:
    return _pwd_hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return _pwd_hasher.verify(plain, hashed)
    except Exception:
        return False


# 简单强度校验：≥8 位、至少一个字母 + 一个数字；可按需加强（特殊字符/黑名单）
_PWD_RE = re.compile(r"^(?=.*[A-Za-z])(?=.*\d).{8,}$")


def validate_password_strength(plain: str) -> None:
    """弱密码抛 ValueError，service 层转换为 HTTPException 400。"""
    if not _PWD_RE.match(plain):
        raise ValueError("password must be >= 8 chars and contain both letters and digits")


# ============ JWT ============
TokenTypes = Literal["access", "refresh"]


def _create_token(
    *,
    sub: int,
    tenant_id: int,
    token_type: TokenTypes,
    extra_claims: dict[str, Any] | None = None,
    expires_delta: timedelta,
) -> tuple[str, str, datetime]:
    """统一 token 签发。返回 (token, jti, expires_at)。"""
    now = datetime.now(timezone.utc)
    jti = str(uuid.uuid4())
    payload: dict[str, Any] = {
        "sub": str(sub),
        "tenant_id": tenant_id,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": jti,
        "type": token_type,
    }
    if extra_claims:
        payload.update(extra_claims)
    token = jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)
    return token, jti, now + expires_delta


def create_access_token(
    *, sub: int, tenant_id: int, roles: list[str], scopes: list[str]
) -> tuple[str, str, datetime]:
    return _create_token(
        sub=sub,
        tenant_id=tenant_id,
        token_type="access",
        extra_claims={"roles": roles, "scopes": scopes},
        expires_delta=timedelta(minutes=settings.access_token_expire_minutes),
    )


def create_refresh_token(*, sub: int, tenant_id: int) -> tuple[str, str, datetime]:
    return _create_token(
        sub=sub,
        tenant_id=tenant_id,
        token_type="refresh",
        expires_delta=timedelta(days=settings.refresh_token_expire_days),
    )


def decode_token(token: str) -> dict[str, Any]:
    """校验签名 + exp；其余业务校验（type/jti 黑名单）由调用方负责。"""
    return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])


# ============ token_hash（DB 存 SHA256，防原值泄漏）============
def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
