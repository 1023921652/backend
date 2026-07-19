"""Redis 维护的 access token jti 黑名单（方案 A）。

策略：
- logout 时把 access token 的 jti 写入 Redis，TTL = 剩余有效期；token 过期时黑名单条目同步清理
- get_current_user 每次请求查一次 EXISTS，O(1)
- Redis 故障 fail-open（放行 + warning log），避免 Redis 抖动打挂所有需认证接口

键命名空间：auth:blk:{jti}（与 LangGraph checkpoints:* / rate limiter 隔离）。
"""
from __future__ import annotations

import logging

import redis.asyncio as redis

from app.agent.config.redis_config import pool

logger = logging.getLogger(__name__)

# 共享现有连接池，但 decode_responses=True 拿 str 结果方便处理；
# 不影响 LangGraph 的 state_redis_client（它必须保持 bytes）
_blocklist = redis.Redis(connection_pool=pool, decode_responses=True)

_KEY_PREFIX = "auth:blk:"


async def revoke_access_jti(jti: str, remaining_ttl_seconds: int) -> None:
    """撤销一个 access token。TTL=剩余有效期，到期自动清理。"""
    if remaining_ttl_seconds <= 0:
        return  # token 已过期，没必要写
    try:
        await _blocklist.setex(f"{_KEY_PREFIX}{jti}", remaining_ttl_seconds, "1")
    except Exception:
        logger.exception("redis blocklist write failed; jti=%s", jti)


async def is_access_jti_revoked(jti: str) -> bool:
    """查询 jti 是否已撤销。Redis 故障时 fail-open 返回 False。"""
    try:
        return bool(await _blocklist.exists(f"{_KEY_PREFIX}{jti}"))
    except Exception:
        logger.warning("redis blocklist lookup failed; fail-open", exc_info=True)
        return False
