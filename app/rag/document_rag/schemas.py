"""Milvus 集合 schema 与索引定义。

两个集合：
- chapter_collection：物理存储 chapter + document 元数据，挂 BM25 函数（sparse_vector），
  字段索引 INVERTED（document_id / chapter_id / titles）方便反查。
- sentence_collection：dense vector HNSW，是父子查询的一级召回入口。

`ensure_collections(client)` 幂等：已存在则跳过（支持增量插入），不存在则按 schema 创建。
schema 升级时需手动 drop_collection 重建。
"""
from __future__ import annotations

import logging

from pymilvus import DataType, Function, FunctionType

from app.rag.document_rag.config import (
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
    schema = _new_schema()
    schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field("chapter_id", DataType.INT64)
    schema.add_field("document_id", DataType.INT64)
    schema.add_field("document_title", DataType.VARCHAR, max_length=512)
    schema.add_field("chapter_title", DataType.VARCHAR, max_length=512)
    # language 必须在 chapter_text 之前：multi_analyzer_params 的 by_field="language"
    # 引用本字段，官网示例顺序也是如此（被引用字段先定义）。
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
        # enable_match=True,
        multi_analyzer_params=MULTI_LANG_ANALYZER_PARAMS,
    )
    schema.add_field("char_count", DataType.INT64)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)

    # chapter_text -> sparse_vector（BM25）。函数挂 schema 后由 milvus 自动计算。
    schema.add_function(
        Function(
            name="chap_text_bm25_emb",
            input_field_names=["chapter_text"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )
    return schema


def _build_sentence_schema():
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
    return schema


def _new_schema():
    # pymilvus MilvusClient.create_schema 在 2.5+ 不再需要传 args；保留兼容
    from pymilvus import MilvusClient

    return MilvusClient.create_schema(
        enable_dynamic_field=True
    )


def _chapter_index_params(client):
    params = client.prepare_index_params()
    # AUTOINDEX 不接受自定义 params（只允许 metric_type），所以用 SPARSE_INVERTED_INDEX
    # 才能调 inverted_index_algo / bm25_k1 / bm25_b。如不需要调优可换回 AUTOINDEX 并删 params。
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
    """幂等创建 chapter + sentence 集合；已存在则跳过。"""
    chapter_ok = client.has_collection(CHAPTER_COLL)
    sentence_ok = client.has_collection(SENTENCE_COLL)
    if chapter_ok and sentence_ok:
        logger.info(
            "collections already exist: %s, %s (incremental insert)",
            CHAPTER_COLL,
            SENTENCE_COLL,
        )
        return

    if not chapter_ok:
        logger.info("creating collection: %s", CHAPTER_COLL)
        client.create_collection(
            collection_name=CHAPTER_COLL,
            schema=_build_chapter_schema(),
            index_params=_chapter_index_params(client),
            consistency_level=CONSISTENCY_LEVEL,
        )

    if not sentence_ok:
        logger.info("creating collection: %s", SENTENCE_COLL)
        client.create_collection(
            collection_name=SENTENCE_COLL,
            schema=_build_sentence_schema(),
            index_params=_sentence_index_params(client),
            consistency_level=CONSISTENCY_LEVEL,
        )

    logger.info("collections ready")
