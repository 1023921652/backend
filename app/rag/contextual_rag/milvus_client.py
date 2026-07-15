"""MilvusClient 进程级单例（contextual_rag 独立持有）。

与 document_rag 的单例是两个独立对象，但底层连接到同一个 Milvus 实例。
分两个单例是为了：未来如果 contextual_rag 要切到不同 Milvus 实例，
改这里的 env 即可，不影响 document_rag。

pymilvus MilvusClient 内部维护连接，重复 new 不会泄漏资源，
但每次 new 会建立新 transport；FastAPI 多请求复用同一实例更高效。
"""
from __future__ import annotations

import logging
import threading

from pymilvus import MilvusClient

from app.rag.contextual_rag.config import MILVUS_TOKEN, MILVUS_URI

logger = logging.getLogger(__name__)

_client: MilvusClient | None = None
_lock = threading.Lock()


def get_milvus_client() -> MilvusClient:
    """返回进程级单例 MilvusClient；线程安全懒加载。"""
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            logger.info(
                "[contextual_rag] connecting to Milvus: %s", MILVUS_URI
            )
            _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            logger.info("[contextual_rag] Milvus client ready")
    return _client
