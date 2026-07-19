"""Redis 连接池（共享）。

显式创建池以控制最大连接数、复用 TCP keepalive / 健康检查。
所有客户端（state_redis_client / str_redis_client）共用此池。
"""
from __future__ import annotations

import redis.asyncio as redis

from app.core.config import settings

pool = redis.ConnectionPool(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    max_connections=settings.redis_max_connections,
    health_check_interval=settings.redis_health_check_interval,  # 每 30s PING 检查连接存活
    socket_keepalive=True,  # 开启底层 TCP Keep-Alive
)
