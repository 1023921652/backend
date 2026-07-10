import json
from typing import Any
import redis.asyncio as redis

REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0

# ==========================================
# 1. 显式创建连接池 (推荐做法)
# ==========================================
# 显式创建连接池可以让你控制最大连接数(max_connections)，保护 Redis 服务
pool = redis.ConnectionPool(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=REDIS_DB,
    max_connections=100,  # 生产环境建议设置上限，视服务器配置而定
    # 🌟 新增下面这两个参数（生产环境必备！）
    health_check_interval=30,  # 每隔 30 秒向 Redis 发送一个 PING 检查连接是否存活
    socket_keepalive=True      # 开启底层的 TCP Keep-Alive
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
            checkpoint_prefix='checkpoints',
            ttl={
                    "default_ttl": 60,       # 60 分钟 (即 1 小时)
                    "refresh_on_read": True  # 每次读取/交互时，自动重置倒计时
                }
        )
        await checkpointer.asetup()
        return checkpointer
    except Exception as e:
        print(e)









