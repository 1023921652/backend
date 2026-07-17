"""RAG REST 接口：插入 / 父子查询 / 列表 / 删除。

依赖 service 层；milvus 连接异常上抛为 HTTPException 503，
避免单接口故障影响整应用启动（agent 注入只取函数引用，不立即连接）。

所有 endpoint 全 async，走 AsyncMilvusClient + asyncio event loop，
避免 starlette threadpool（默认 40 worker）成为高并发瓶颈。
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request

from app.core.rate_limit import (
    RATE_LIMIT_INGEST_PER_MIN,
    RATE_LIMIT_SEARCH_PER_MIN,
    limiter,
)
from app.rag.document_rag import service
from app.rag.document_rag.milvus_client import get_async_milvus_client
from app.rag.document_rag.schemas import aensure_collections
from app.schemas.rag_types import (
    CollectionInfo,
    DeleteCollectionsRequest,
    DeleteCollectionsResult,
    DeleteDocumentsRequest,
    DeleteStats,
    DocumentInput,
    DocumentSummary,
    IngestStats,
    SearchRequest,
    SearchResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/rag", tags=["rag"])


async def _ensure_ready() -> None:
    """接口层调用前确保集合存在；milvus 不可达时统一转 503。"""
    try:
        client = await get_async_milvus_client()
        await aensure_collections(client)
    except Exception as e:
        logger.exception("milvus not ready")
        raise HTTPException(
            status_code=503,
            detail=f"Milvus unavailable: {type(e).__name__}: {e}",
        )


@router.post("/documents", response_model=IngestStats)
@limiter.limit(f"{RATE_LIMIT_INGEST_PER_MIN}/minute")
async def ingest_documents(request: Request, documents: list[DocumentInput]) -> IngestStats:
    """批量插入 chapter + 切分后的 sentence chunk。"""
    if not documents:
        return IngestStats(inserted_chapters=0, inserted_sentences=0)
    await _ensure_ready()
    try:
        return await service.aingest_documents(documents)
    except Exception as e:
        logger.exception("ingest failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/search", response_model=list[SearchResult])
@limiter.limit(f"{RATE_LIMIT_SEARCH_PER_MIN}/minute")
async def search(request: Request, req: SearchRequest) -> list[SearchResult]:
    """父子查询：sentence 召回 → 聚合 doc/chapter → top 3 docs。
        "top_k"未使用，当前由环境变量SEARCH_LIMIT控制
    """
    await _ensure_ready()
    try:
        return await service.ahierarchical_search(req.query)
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.post("/search/fulltext", response_model=list[SearchResult])
@limiter.limit(f"{RATE_LIMIT_SEARCH_PER_MIN}/minute")
async def search_fulltext(request: Request, req: SearchRequest) -> list[SearchResult]:
    """BM25 关键词全文检索：直接搜 chapter_collection 的 sparse_vector。

    适合精确术语、型号、ID、专有名词词面匹配；与 /v1/rag/search（向量语义检索）
    互补。响应 retrieval_mode 字段值为 '[fulltext] ...' 区分；
    sentence_hits 字段为空（fulltext 无 chunk 级命中）。
    """
    await _ensure_ready()
    try:
        return await service.afulltext_search(req.query)
    except Exception as e:
        logger.exception("fulltext search failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/documents", response_model=list[DocumentSummary])
async def list_documents() -> list[DocumentSummary]:
    """列出所有 document（去重聚合 chapter 维度）。"""
    await _ensure_ready()
    try:
        return await service.alist_documents()
    except Exception as e:
        logger.exception("list failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.delete("/documents", response_model=list[DeleteStats])
async def delete_documents(req: DeleteDocumentsRequest) -> list[DeleteStats]:
    """批量按 document_id 级联删除 sentence + chapter。

    不存在的 id 返回 0/0 占位；单项失败不中断其余项。
    """
    await _ensure_ready()
    try:
        return await service.adelete_documents(req.document_ids)
    except Exception as e:
        logger.exception("batch delete documents failed")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@router.get("/collections", response_model=list[CollectionInfo])
async def list_collections() -> list[CollectionInfo]:
    """列出 milvus 实例上所有集合（含行数与 RAG 标识）。

    集合级查询：不调 ensure_collections，允许在 schema 未初始化时查看现状。
    """
    try:
        return await service.alist_collections()
    except Exception as e:
        logger.exception("list collections failed")
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {type(e).__name__}: {e}")


@router.delete("/collections", response_model=DeleteCollectionsResult)
async def delete_collections(req: DeleteCollectionsRequest) -> DeleteCollectionsResult:
    """批量删除任意集合。允许任意集合名，前端需二次确认。

    删除 RAG 集合后，下次插入数据时 ensure_collections 会自动按 schema 重建；
    但删除后到下次插入之间，search/agent 调用会失败。
    """
    try:
        return await service.adelete_collections(req.collection_names)
    except Exception as e:
        logger.exception("delete collections failed")
        raise HTTPException(status_code=503, detail=f"Milvus unavailable: {type(e).__name__}: {e}")
