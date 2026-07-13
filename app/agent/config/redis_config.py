"""Redis 连接池 + LangGraph checkpointer 工厂。

所有连接参数从环境变量读（.env 已由 app.main load_dotenv 加载），
未设置时使用合理默认值，方便本地开发。

注意：state_redis_client 不能加 decode_responses=True，LangGraph 需要 bytes。
"""
import os

import redis.asyncio as redis

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_MAX_CONNECTIONS = int(os.getenv("REDIS_MAX_CONNECTIONS", "100"))
REDIS_HEALTH_CHECK_INTERVAL = int(os.getenv("REDIS_HEALTH_CHECK_INTERVAL", "30"))

# ==========================================
# 1. 显式创建连接池 (推荐做法)
# ==========================================
# 显式创建连接池可以让你控制最大连接数(max_connections)，保护 Redis 服务
pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    max_connections=REDIS_MAX_CONNECTIONS,
    health_check_interval=REDIS_HEALTH_CHECK_INTERVAL,  # 每 30s PING 检查连接存活
    socket_keepalive=True,  # 开启底层 TCP Keep-Alive
)
# ==========================================
# 2. 实例化客户端 (复用同一个池子)
# ==========================================

# A. 专门给 LangGraph Checkpointer 用的客户端
# ！！！千万不要加 decode_responses=True，让它保持处理 bytes 格式！！！
state_redis_client = redis.Redis(connection_pool=pool)


from langgraph.checkpoint.redis.aio import AsyncRedisSaver


async def get_redis_checkpointer():
    try:
        checkpointer = AsyncRedisSaver(
            redis_client=state_redis_client,
            checkpoint_prefix=os.getenv("REDIS_CHECKPOINT_PREFIX", "checkpoints"),
            ttl={
                "default_ttl": int(os.getenv("REDIS_CHECKPOINT_TTL_MINUTES", "60")),
                "refresh_on_read": True,
            },
        )
        await checkpointer.asetup()
        return checkpointer
    except Exception as e:
        print(e)