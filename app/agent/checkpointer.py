"""LangGraph Redis checkpointer 工厂。

依赖 app.db.redis.clients.state_redis_client（bytes 模式）。
构造失败返回 None，agent 退化为无状态。
"""
from __future__ import annotations

import logging

from langgraph.checkpoint.redis.aio import AsyncRedisSaver

from app.core.config import settings
from app.db.redis.clients import state_redis_client

logger = logging.getLogger(__name__)


async def get_redis_checkpointer():
    try:
        checkpointer = AsyncRedisSaver(
            redis_client=state_redis_client,
            checkpoint_prefix=settings.redis_checkpoint_prefix,
            ttl={
                "default_ttl": settings.redis_checkpoint_ttl_minutes,
                "refresh_on_read": True,
            },
        )
        await checkpointer.asetup()
        return checkpointer
    except Exception:
        logger.exception("redis checkpointer setup failed; agent will be stateless")
        return None
