"""Contextual RAG REST 接口：插入 / 父子查询 / 全文检索 / 列表 / 删除。

与 app/api/v1/rag.py 结构镜像，prefix 改为 /v1/contextual_rag，
底层调 app.rag.contextual_rag.service。集合 schema 与原 RAG 几乎一致
（仅 sentence_collection 多一个 context 字段，由 service 层透明填充）。

降级：milvus 不可达时接口返回 503，不影响 /v1/rag/* 与 agent 启动。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.rag.contextual_rag import service
from app.rag.contextual_rag.milvus_client import get_milvus_client
from app.rag.contextual_rag.schemas import ensure_collections
from app.schemas.rag_types import (
    CollectionInfo,
    ContextualSearchResult,
    DeleteCollectionsRequest,
    DeleteCollectionsResult,
    DeleteDocumentsRequest,
    DeleteStats,
    DocumentInput,
    DocumentSummary,
    IngestStats,
    SearchRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/contextual_rag", tags=["contextual_rag"])


def _ensure_ready() -> None:
    """接口层调用前确保集合存在；milvus 不可达时统一转 503。"""
    try:
        client = get_milvus_client()
        ensure_collections(client)
    except Exception as e:
        logger.exception("[contextual_rag] milvus not ready")
        raise HTTPException(
            status_code=503,
            detail=f"Milvus unavailable: {type(e).__name__}: {e}",
        )


@router.post("/documents", response_model=IngestStats)
def ingest_documents(documents: list[DocumentInput]) -> IngestStats:
    """批量插入 chapter + 切分后的 sentence chunk（每个 chunk 附带 LLM 生成的上下文摘要）。

    成本：每个 chunk 一次 LLM 调用（生成上下文），与 chunk 数量线性相关。
    """
    if not documents:
        return IngestStats(inserted_chapters=0, inserted_sentences=0)
    _ensure_ready()
    try:
        return service.ingest_documents(documents)
    except Exception as e:
        logger.exception("[contextual_rag] ingest failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/search", response_model=list[ContextualSearchResult])
def search(req: SearchRequest) -> list[ContextualSearchResult]:
    """父子查询：sentence 召回（含 context 字段） → 聚合 doc/chapter → top N docs。

    返回 ContextualSearchResult 列表：sentence_hits 内每项带 LLM 生成的 context 字段。
    """
    _ensure_ready()
    try:
        return service.hierarchical_search(req.query)
    except Exception as e:
        logger.exception("[contextual_rag] search failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/search/fulltext", response_model=list[ContextualSearchResult])
def search_fulltext(req: SearchRequest) -> list[ContextualSearchResult]:
    """BM25 关键词全文检索：直接搜 contextual_chapter_collection 的 sparse_vector。

    与 /search（dense 语义检索）互补。fulltext 无 chunk 级命中，
    因此返回的 ContextualSearchResult.sentence_hits 为空（看不到 context 字段）。
    """
    _ensure_ready()
    try:
        return service.fulltext_search(req.query)
    except Exception as e:
        logger.exception("[contextual_rag] fulltext search failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/documents", response_model=list[DocumentSummary])
def list_documents() -> list[DocumentSummary]:
    """列出所有 document（去重聚合 chapter 维度）。"""
    _ensure_ready()
    try:
        return service.list_documents()
    except Exception as e:
        logger.exception("[contextual_rag] list failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.delete("/documents", response_model=list[DeleteStats])
def delete_documents(req: DeleteDocumentsRequest) -> list[DeleteStats]:
    """批量按 document_id 级联删除 sentence + chapter。"""
    _ensure_ready()
    try:
        return service.delete_documents(req.document_ids)
    except Exception as e:
        logger.exception("[contextual_rag] batch delete documents failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/collections", response_model=list[CollectionInfo])
def list_collections() -> list[CollectionInfo]:
    """列出 milvus 实例上所有集合（含行数与 contextual RAG 标识）。"""
    try:
        return service.list_collections()
    except Exception as e:
        logger.exception("[contextual_rag] list collections failed")
        raise HTTPException(
            status_code=503,
            detail=f"Milvus unavailable: {type(e).__name__}: {e}",
        )


@router.delete("/collections", response_model=DeleteCollectionsResult)
def delete_collections(req: DeleteCollectionsRequest) -> DeleteCollectionsResult:
    """批量删除任意集合。删除 contextual RAG 集合后，下次插入数据时
    ensure_collections 会自动按 schema 重建。"""
    try:
        return service.delete_collections(req.collection_names)
    except Exception as e:
        logger.exception("[contextual_rag] delete collections failed")
        raise HTTPException(
            status_code=503,
            detail=f"Milvus unavailable: {type(e).__name__}: {e}",
        )
