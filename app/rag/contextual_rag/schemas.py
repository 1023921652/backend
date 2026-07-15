"""Contextual RAG 的 Milvus 集合 schema 与索引定义。

两个集合：
- contextual_chapter_collection：与 document_rag 的 chapter_collection 完全一致
  （字段、索引、analyzer、BM25 function 全部相同）。
- contextual_sentence_collection：在 document_rag sentence_collection 基础上
  增加 `context VARCHAR(1024)` 字段（存 LLM 生成的上下文摘要，无 analyzer、
  不参与 BM25），其余字段、HNSW 索引、INVERTED 索引一致。

ensure_collections(client) 幂等：已存在则跳过，不存在则按 schema 创建。
"""
from __future__ import annotations

import logging

from pymilvus import DataType, Function, FunctionType

from app.rag.contextual_rag.config import (
    CHAPTER_COLL,
    CONSISTENCY_LEVEL,
    DEFAULT_LANGUAGE,
    INDEX_HNSW_EF,
    INDEX_HNSW_M,
    INDEX_TYPE,
    MULTI_LANG_ANALYZER_PARAMS,
    SENTENCE_COLL,
)
from app.rag.embedding import VECTOR_DIM

logger = logging.getLogger(__name__)


def _build_chapter_schema():
    """与 document_rag 的 chapter schema 完全一致。"""
    schema = _new_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("chapter_id", DataType.INT64)
    schema.add_field("document_id", DataType.INT64)
    schema.add_field("document_title", DataType.VARCHAR, max_length=512)
    schema.add_field("chapter_title", DataType.VARCHAR, max_length=512)
    schema.add_field(
        field_name="language",
        datatype=DataType.VARCHAR,
        max_length=255,
        default_value=DEFAULT_LANGUAGE,
    )
    schema.add_field(
        "chapter_text",
        DataType.VARCHAR,
        max_length=16384,
        enable_analyzer=True,
        multi_analyzer_params=MULTI_LANG_ANALYZER_PARAMS,
    )
    schema.add_field("char_count", DataType.INT64)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(
        Function(
            name="ctx_chap_text_bm25_emb",
            input_field_names=["chapter_text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def _build_sentence_schema():
    """在 document_rag sentence schema 基础上加 context 字段。"""
    schema = _new_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("document_id", DataType.INT64)
    schema.add_field("chapter_id", DataType.INT64)
    schema.add_field("chunk_text", DataType.VARCHAR, max_length=2048)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
    schema.add_field("chunk_index", DataType.INT64)
    schema.add_field("char_count", DataType.INT64)
    schema.add_field("document_title", DataType.VARCHAR, max_length=512)
    schema.add_field("chapter_title", DataType.VARCHAR, max_length=512)
    # 【新字段】LLM 生成的上下文摘要；仅持久化供检索结果返回，无 analyzer、无索引。
    schema.add_field("context", DataType.VARCHAR, max_length=1024)
    return schema


def _new_schema():
    from pymilvus import MilvusClient

    return MilvusClient.create_schema(enable_dynamic_field=True)


def _chapter_index_params(client):
    params = client.prepare_index_params()
    params.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={
            "inverted_index_algo": "DAAT_MAXSCORE",
            "bm25_k1": 1.2,
            "bm25_b": 0.75,
        },
    )
    params.add_index(field_name="document_id", index_type="INVERTED")
    params.add_index(field_name="chapter_id", index_type="INVERTED")
    params.add_index(field_name="document_title", index_type="INVERTED")
    params.add_index(field_name="chapter_title", index_type="INVERTED")
    return params


def _sentence_index_params(client):
    params = client.prepare_index_params()
    params.add_index(
        field_name="dense_vector",
        index_type=INDEX_TYPE,
        metric_type="COSINE",
        params={"M": INDEX_HNSW_M, "efConstruction": INDEX_HNSW_EF},
    )
    params.add_index(field_name="document_id", index_type="INVERTED")
    params.add_index(field_name="chapter_id", index_type="INVERTED")
    params.add_index(field_name="document_title", index_type="INVERTED")
    params.add_index(field_name="chapter_title", index_type="INVERTED")
    params.add_index(field_name="chunk_index", index_type="STL_SORT")
    return params


def ensure_collections(client) -> None:
    """幂等创建 contextual chapter + sentence 集合；已存在则跳过。"""
    chapter_ok = client.has_collection(CHAPTER_COLL)
    sentence_ok = client.has_collection(SENTENCE_COLL)
    if chapter_ok and sentence_ok:
        logger.info(
            "[contextual_rag] collections already exist: %s, %s (incremental insert)",
            CHAPTER_COLL,
            SENTENCE_COLL,
        )
        return

    if not chapter_ok:
        logger.info("[contextual_rag] creating collection: %s", CHAPTER_COLL)
        client.create_collection(
            collection_name=CHAPTER_COLL,
            schema=_build_chapter_schema(),
            index_params=_chapter_index_params(client),
            consistency_level=CONSISTENCY_LEVEL,
        )

    if not sentence_ok:
        logger.info("[contextual_rag] creating collection: %s", SENTENCE_COLL)
        client.create_collection(
            collection_name=SENTENCE_COLL,
            schema=_build_sentence_schema(),
            index_params=_sentence_index_params(client),
            consistency_level=CONSISTENCY_LEVEL,
        )

    logger.info("[contextual_rag] collections ready")
