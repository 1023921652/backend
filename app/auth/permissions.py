"""权限 scope 常量与角色模板。

scope 格式固定为 `resource:action`，精确匹配不做通配。
PERMISSIONS 是全局 scope 字典（首次启动时 upsert 到 permissions 表）。
ROLE_TEMPLATES 定义 builtin 角色（owner / admin / member）拥有的 scope 集合。
"""
from __future__ import annotations

# ============ 全局权限字典 ============
# (resource, action) → scope 字符串
PERMISSIONS: dict[str, tuple[str, str]] = {
    "enterprise:read": ("enterprise", "read"),
    "enterprise:create": ("enterprise", "create"),
    "enterprise:update": ("enterprise", "update"),
    "enterprise:delete": ("enterprise", "delete"),
    "user:read": ("user", "read"),
    "user:create": ("user", "create"),
    "user:update": ("user", "update"),
    "user:delete": ("user", "delete"),
    "role:read": ("role", "read"),
    "role:create": ("role", "create"),
    "role:update": ("role", "update"),
    "role:delete": ("role", "delete"),
    "member:read": ("member", "read"),
    "member:invite": ("member", "invite"),
    "member:update": ("member", "update"),
    "member:remove": ("member", "remove"),
    "permission:read": ("permission", "read"),
    "audit:read": ("audit", "read"),
}

ALL_SCOPES: frozenset[str] = frozenset(PERMISSIONS.keys())


def read_only_scopes() -> frozenset[str]:
    """所有 *:read scope（member 角色基线）。"""
    return frozenset(s for s in ALL_SCOPES if s.endswith(":read"))


# ============ 内置角色模板 ============
# 创建企业时自动产出 owner / admin / member 三个 builtin 角色
ROLE_TEMPLATES: dict[str, frozenset[str]] = {
    "owner": ALL_SCOPES,
    "admin": ALL_SCOPES - {"enterprise:delete"},
    "member": read_only_scopes(),
}

# 标记为 builtin 的角色名（不允许删除 / 改名 / 改 scopes）
BUILTIN_ROLE_NAMES: frozenset[str] = frozenset(ROLE_TEMPLATES.keys())
