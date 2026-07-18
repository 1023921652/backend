"""异步 RAG 路径的基础测试。

不依赖真实 Milvus / Redis（用 TestClient 打 endpoint，503 也算通过——证明
async 路径正确接线，milvus 不可达时优雅降级）。

依赖真实 Milvus 的端到端测试见手动验证步骤。
"""
from __future__ import annotations

import inspect

import pytest
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_rag_tools_registered_as_async():
    """@tool 函数应被 langchain 注册为 StructuredTool 且 .coroutine 已设置。"""
    from app.rag.document_rag.tools import (
        rag_decomposed_search,
        rag_fulltext_search,
        rag_simple_search,
    )

    for t in (rag_simple_search, rag_fulltext_search, rag_decomposed_search):
        assert hasattr(t, "coroutine"), f"{t.name} missing .coroutine (not async)"
        assert inspect.iscoroutinefunction(t.coroutine), (
            f"{t.name}.coroutine must be a coroutine function"
        )


def test_rag_service_has_async_variants():
    """service 层应有所有 a* 前缀的 async 函数。"""
    from app.rag.document_rag import service

    expected = [
        "aingest_documents",
        "ahierarchical_search",
        "afulltext_search",
        "alist_documents",
        "adelete_document",
        "adelete_documents",
        "alist_collections",
        "adelete_collections",
        "_aaggregate_to_doc_results",
    ]
    for name in expected:
        fn = getattr(service, name, None)
        assert fn is not None, f"service.{name} missing"
        assert inspect.iscoroutinefunction(fn), f"service.{name} not async"


def test_repository_has_async_mirrors():
    """repository 层 12 个 sync 函数都应有 async 兄弟。"""
    from app.rag.document_rag import repository as repo

    expected_async = [
        "ainsert_chapters",
        "ainsert_sentences",
        "asearch_sentences",
        "aquery_chapters_by_document",
        "asearch_chapters_by_bm25",
        "aquery_all_chapters",
        "adelete_chapters_by_document",
        "adelete_sentences_by_document",
        "alist_collections",
        "ahas_collection",
        "aget_collection_row_count",
        "adrop_collection",
    ]
    for name in expected_async:
        fn = getattr(repo, name, None)
        assert fn is not None, f"repo.{name} missing"
        assert inspect.iscoroutinefunction(fn), f"repo.{name} not async"


def test_async_milvus_singleton_exists():
    """async 单例 getter 应存在且为协程。"""
    from app.rag.document_rag.milvus_client import get_async_milvus_client

    assert inspect.iscoroutinefunction(get_async_milvus_client)


def test_collections_endpoint_does_not_require_milvus():
    """/v1/rag/collections 不调 ensure_collections，但 milvus 不可达时应 503。

    证明 async endpoint 接线完整：能进 endpoint、调 service、识别异常、返回 503。
    """
    resp = client.get("/v1/rag/collections")
    # 没有 milvus 时 503；有 milvus 时 200。两者都说明 async 路径打通。
    assert resp.status_code in (200, 503)


def test_ingest_empty_body_short_circuits_before_milvus():
    """空 body 走早返分支，不触发 milvus；返回 200 + 0/0 stats。

    注意：rate limit 装饰器仍生效；测试若反复调用需让计数器重置。
    """
    resp = client.post("/v1/rag/documents", json=[])
    assert resp.status_code in (200, 429)  # 429 = 触发限流（其它测试累积计数）
    if resp.status_code == 200:
        body = resp.json()
        assert body["inserted_chapters"] == 0
        assert body["inserted_sentences"] == 0
