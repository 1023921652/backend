"""进程级并发治理：asyncio.Semaphore 集中管理。

为什么需要：
- DashScope / DeepSeek 等 LLM 服务有 QPS / 并发上限；突发请求会触发 429。
- Milvus 单次大批量 insert 也会拖慢；并发 insert 需要限速。
- async 化后，没有 Semaphore 的话 asyncio.gather(N) 会同时发出 N 个请求。

设计：
- 每个 Semaphore 进程级、模块加载时构造；env var 控制上限。
- 业务调用点 `async with LLM_SEMAPHORE: await llm.ainvoke(...)`。
- k8s 多 pod 下，每个 pod 独立计数——若 DashScope 全局上限 M，部署 N pod，
  每 pod 配 M/N 并发；这层不解决跨 pod 协调（需要分布式限流）。
"""
from __future__ import annotations

import asyncio
import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """读 env int；非法值告警并回退默认。"""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        v = int(raw)
        if v <= 0:
            logger.warning("%s=%r must be positive; fallback to %d", name, raw, default)
            return default
        return v
    except ValueError:
        logger.warning("%s=%r not int; fallback to %d", name, raw, default)
        return default


# LLM 全局并发上限：contextualizer / rag_decomposed / task 直 LLM 共享
# 当前未在 service 层强制使用（避免侵入本期范围），供后续接入
LLM_MAX_CONCURRENCY = _env_int("LLM_MAX_CONCURRENCY", 16)
LLM_SEMAPHORE = asyncio.Semaphore(LLM_MAX_CONCURRENCY)

# Milvus 批量 insert 并发上限（同 pod 内多请求同时 insert 时限速）
MILVUS_INSERT_CONCURRENCY = _env_int("MILVUS_INSERT_CONCURRENCY", 4)
MILVUS_INSERT_SEMAPHORE = asyncio.Semaphore(MILVUS_INSERT_CONCURRENCY)

logger.info(
    "concurrency governance: llm_max=%d milvus_insert_max=%d",
    LLM_MAX_CONCURRENCY, MILVUS_INSERT_CONCURRENCY,
)
