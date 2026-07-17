"""MilvusClient 进程级单例（同步 + 异步两套）。

pymilvus MilvusClient 内部维护连接，重复 new 不会造成资源泄漏，
但每次 new 会建立新 transport；FastAPI 多请求复用同一实例更高效。

两套并存：
- 同步 `get_milvus_client()`：脚本 / 同步 service 函数 / 测试用
- 异步 `get_async_milvus_client()`：FastAPI async endpoint / async service 用

AsyncMilvusClient 在 pymilvus 3.0+ 提供，API 与 sync 版几乎一致，
返回结构相同；只是所有 IO 方法都是协程，需要在 event loop 里 await。
"""
from __future__ import annotations

import asyncio
import logging
import threading

from pymilvus import AsyncMilvusClient, MilvusClient

from app.rag.document_rag.config import MILVUS_TOKEN, MILVUS_URI

logger = logging.getLogger(__name__)

# ============ 同步单例 ============
_client: MilvusClient | None = None
_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """返回进程级同步单例 MilvusClient；线程安全懒加载。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            logger.info("connecting to Milvus (sync): %s", MILVUS_URI)
            _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            logger.info("sync Milvus client ready")
    return _client


# ============ 异步单例 ============
_async_client: AsyncMilvusClient | None = None
_async_lock: asyncio.Lock = asyncio.Lock()


async def get_async_milvus_client() -> AsyncMilvusClient:
    """返回进程级异步单例 AsyncMilvusClient；event loop 内双检锁懒加载。

    注意：`asyncio.Lock` 绑定到当前 event loop。FastAPI / uvicorn 单 worker
    下 event loop 与进程同生命周期，无问题；若运行期 event loop 替换（测试场景），
    需要重置 `_async_client=None` + 重建 lock。
    """
    global _async_client
    if _async_client is not None:
        return _async_client
    async with _async_lock:
        if _async_client is None:
            logger.info("connecting to Milvus (async): %s", MILVUS_URI)
            _async_client = AsyncMilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            logger.info("async Milvus client ready")
    return _async_client
