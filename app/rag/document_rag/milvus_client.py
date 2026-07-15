"""MilvusClient 进程级单例。

pymilvus MilvusClient 内部维护连接，重复 new 不会造成资源泄漏，
但每次 new 会建立新 transport；FastAPI 多请求复用同一实例更高效。
"""
from __future__ import annotations

import logging
import threading

from pymilvus import MilvusClient

from app.rag.document_rag.config import MILVUS_TOKEN, MILVUS_URI

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
            logger.info("connecting to Milvus: %s", MILVUS_URI)
            _client = MilvusClient(uri=MILVUS_URI, token=MILVUS_TOKEN)
            logger.info("Milvus client ready")
    return _client
