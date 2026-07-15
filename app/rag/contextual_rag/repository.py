"""数据访问层：每个函数封装一次 milvus 调用。

与 app.rag.document_rag.repository 结构一致，区别：
- 集合名指向 contextual_rag.config 的新集合
- search_sentences 的 output_fields 增加 context 字段
"""
from __future__ import annotations

import logging
from typing import Any

from pymilvus import MilvusClient

from app.rag.contextual_rag.config import CHAPTER_COLL, SENTENCE_COLL

logger = logging.getLogger(__name__)


# ============ insert ============
def insert_chapters(client: MilvusClient, rows: list[dict]) -> int:
    if not rows:
        return 0
    client.insert(collection_name=CHAPTER_COLL, data=rows)
    return len(rows)


def insert_sentences(client: MilvusClient, rows: list[dict]) -> int:
    if not rows:
        return 0
    client.insert(collection_name=SENTENCE_COLL, data=rows)
    return len(rows)


# ============ query / search ============
def search_sentences(
    client: MilvusClient,
    query_vector: list[float],
    limit: int,
) -> list[dict[str, Any]]:
    """dense vector 检索 sentence；output_fields 含 context 字段。"""
    res = client.search(
        collection_name=SENTENCE_COLL,
        data=[query_vector],
        anns_field="dense_vector",
        search_params={"metric_type": "COSINE"},
        limit=limit,
        output_fields=[
            "document_id",
            "chapter_id",
            "chunk_text",
            "chunk_index",
            "document_title",
            "chapter_title",
            "context",
        ],
    )
    if not res:
        return []
    return res[0]


def query_chapters_by_document(
    client: MilvusClient,
    document_id: int,
) -> list[dict[str, Any]]:
    """按 document_id 反查所有 chapter 记录。"""
    return client.query(
        collection_name=CHAPTER_COLL,
        filter=f"document_id == {int(document_id)}",
        output_fields=[
            "chapter_id",
            "chapter_title",
            "chapter_text",
            "char_count",
            "document_title",
        ],
    )


def search_chapters_by_bm25(
    client: MilvusClient,
    query_text: str,
    limit: int,
    analyzer_name: str = "default",
) -> list[dict[str, Any]]:
    """BM25 全文检索 contextual_chapter_collection 的 sparse_vector。"""
    res = client.search(
        collection_name=CHAPTER_COLL,
        data=[query_text],
        anns_field="sparse_vector",
        search_params={
            "metric_type": "BM25",
            "analyzer_name": analyzer_name,
        },
        limit=limit,
        output_fields=[
            "document_id",
            "chapter_id",
            "document_title",
            "chapter_title",
            "chapter_text",
            "char_count",
        ],
    )
    if not res:
        return []
    return res[0]


def query_all_chapters(client: MilvusClient) -> list[dict[str, Any]]:
    """全表扫描所有 chapter；用于 list_documents 聚合 document 维度。"""
    return client.query(
        collection_name=CHAPTER_COLL,
        filter="document_id >= 0",
        output_fields=[
            "document_id",
            "document_title",
            "chapter_id",
            "char_count",
        ],
    )


# ============ delete ============
def delete_chapters_by_document(client: MilvusClient, document_id: int) -> None:
    client.delete(
        collection_name=CHAPTER_COLL,
        filter=f"document_id == {int(document_id)}",
    )


def delete_sentences_by_document(client: MilvusClient, document_id: int) -> None:
    client.delete(
        collection_name=SENTENCE_COLL,
        filter=f"document_id == {int(document_id)}",
    )


# ============ collection 管理 ============
def list_collections(client: MilvusClient) -> list[str]:
    return client.list_collections()


def has_collection(client: MilvusClient, name: str) -> bool:
    return client.has_collection(collection_name=name)


def get_collection_row_count(client: MilvusClient, name: str) -> int:
    stats = client.get_collection_stats(collection_name=name)
    return int(stats.get("row_count", 0))


def drop_collection(client: MilvusClient, name: str) -> None:
    client.drop_collection(collection_name=name)
