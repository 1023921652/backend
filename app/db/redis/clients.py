"""Redis 客户端：bytes 模式（LangGraph）+ str 模式（auth blocklist 等）。

**重要**：state_redis_client 绝对不能加 decode_responses=True；
LangGraph checkpointer 反序列化需要 bytes，加了会失败。
"""
from __future__ import annotations

import redis.asyncio as redis

from app.db.redis.pool import pool

# LangGraph checkpointer 专用：bytes 模式
state_redis_client = redis.Redis(connection_pool=pool)

# 业务通用：str 模式（auth blocklist、其他需要 str 返回的场景）
str_redis_client = redis.Redis(connection_pool=pool, decode_responses=True)
