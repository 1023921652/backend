"""业务编排层。

四个公开函数：
- ingest_documents(documents) -> IngestStats   批量插入（去重分组）
- hierarchical_search(query) -> list[SearchResult]   父子查询
- list_documents() -> list[DocumentSummary]   列出所有 document
- delete_document(document_id) -> DeleteStats   级联删除

依赖 repository（milvus 调用）、chunking（切分）、schemas（ensure_collections）、
embedding（dense vector）、config（参数）。

document_id = CRC32(document_title)，跨插入稳定 → 同名 doc 第二次插入是追加 chapter。
chapter_id 在文档内唯一；不同文档可有相同 chapter_id（外键定位用 document_id + chapter_id）。
"""
from __future__ import annotations

import logging
import zlib
from collections import defaultdict

from app.rag.document_rag import repository as repo
from app.rag.document_rag.chunking import chunk_by_sentences
from app.rag.document_rag.config import (
    BM25_SEARCH_LIMIT,
    CHAR_COUNT_THRESHOLD,
    CHAPTER_COLL,
    CHUNK_STEP,
    CHUNK_WINDOW_SIZE,
    RETRIEVAL_MODE_DENSE,
    RETRIEVAL_MODE_FULLTEXT,
    SEARCH_LIMIT,
    SENTENCE_COLL,
    TOP_CHAPTERS,
    TOP_DOCS,
)
from app.rag.document_rag.language import detect_language
from app.rag.document_rag.milvus_client import get_async_milvus_client, get_milvus_client
from app.rag.document_rag.schemas import aensure_collections, ensure_collections
from app.rag.embedding import embeddings
from app.schemas.rag_types import (
    ChapterOut,
    CollectionInfo,
    DeleteCollectionsResult,
    DeleteStats,
    DocumentInput,
    DocumentSummary,
    IngestStats,
    SearchResult,
    SentenceHit,
)

logger = logging.getLogger(__name__)


def _split_title(title: str) -> tuple[str, str]:
    """raw_documents.json 的 title 形如「大文章标题 - 章节标题」。

    拆成 (document_title, chapter_title)；若没有分隔符，chapter_title 复用整段。
    """
    if " - " in title:
        doc, chap = title.split(" - ", 1)
        return doc.strip(), chap.strip()
    return title.strip(), title.strip()


def _make_document_id(document_title: str) -> int:
    """CRC32 生成稳定的 INT64 document_id。"""
    return zlib.crc32(document_title.encode("utf-8"))


# ==========================================
# ingest
# ==========================================
def ingest_documents(documents: list[DocumentInput]) -> IngestStats:
    """批量插入文档。

    1. ensure_collections（首次自动建表，幂等）
    2. 按 document_title 分组、生成稳定 document_id
    3. 写 chapter_collection
    4. 切分 + embed + 写 sentence_collection
    """
    client = get_milvus_client()
    ensure_collections(client)

    # ---- 分组：document_title -> {document_id, chapters: [DocumentInput...]} ----
    grouped: dict[str, dict] = {}
    for doc in documents:
        doc_title, chap_title = _split_title(doc.title)
        document_id = _make_document_id(doc_title)
        grouped.setdefault(doc_title, {"document_id": document_id, "chapters": []})
        grouped[doc_title]["chapters"].append(
            {
                "chapter_id": int(doc.chapter_id),
                "chapter_title": chap_title,
                "paragraphs": list(doc.paragraphs),
            }
        )

    # ---- chapter 行 ----
    chapter_rows: list[dict] = []
    flat_meta: list[dict] = []  # 给 sentence 切分用
    for doc_title, data in grouped.items():
        document_id = data["document_id"]
        for ch in data["chapters"]:
            full_text = "\n".join(ch["paragraphs"])
            language = detect_language(full_text)
            logger.debug(
                "ingest doc=%r chap=%r detected_language=%s",
                doc_title, ch["chapter_title"], language,
            )
            chapter_rows.append(
                {
                    "chapter_id": ch["chapter_id"],
                    "document_id": document_id,
                    "document_title": doc_title,
                    "chapter_title": ch["chapter_title"],
                    "chapter_text": full_text,
                    "language": language,
                    "char_count": len(full_text),
                }
            )
            flat_meta.append(
                {
                    "document_id": document_id,
                    "document_title": doc_title,
                    "chapter_id": ch["chapter_id"],
                    "chapter_title": ch["chapter_title"],
                    "paragraphs": ch["paragraphs"],
                }
            )

    inserted_chapters = repo.insert_chapters(client, chapter_rows)

    # ---- sentence 切分 + embed ----
    sentence_meta: list[dict] = []
    sentence_texts: list[str] = []
    for meta in flat_meta:
        chunks = chunk_by_sentences(
            meta["paragraphs"],
            window_size=CHUNK_WINDOW_SIZE,
            step=CHUNK_STEP,
        )
        for idx, chunk_text in enumerate(chunks):
            sentence_texts.append(chunk_text)
            sentence_meta.append(
                {
                    "chunk_text": chunk_text,
                    "document_id": meta["document_id"],
                    "chapter_id": meta["chapter_id"],
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "document_title": meta["document_title"],
                    "chapter_title": meta["chapter_title"],
                }
            )

    if not sentence_texts:
        return IngestStats(
            inserted_chapters=inserted_chapters,
            inserted_sentences=0,
        )

    logger.info("embedding %d chunks...", len(sentence_texts))
    vectors = embeddings.embed_documents(sentence_texts)

    sentence_rows = []
    for meta, vec in zip(sentence_meta, vectors):
        sentence_rows.append({**meta, "dense_vector": vec})

    inserted_sentences = repo.insert_sentences(client, sentence_rows)

    logger.info(
        "ingest done: chapters=%d sentences=%d",
        inserted_chapters,
        inserted_sentences,
    )
    return IngestStats(
        inserted_chapters=inserted_chapters,
        inserted_sentences=inserted_sentences,
    )


# ==========================================
# hierarchical search
# ==========================================
def hierarchical_search(query: str) -> list[SearchResult]:
    """父子查询：

    1. dense search sentence（chunk 级召回）
    2. 按 document_id 聚合最高分；按 (document_id, chapter_id) 聚合 chapter 最高分
    3. 取 top 3 documents
    4. 字数 < CHAR_COUNT_THRESHOLD → 返回整 doc 的所有 chapter；否则返回 top 2 chapters
    """
    logger.info(
        "hierarchical_search START query=%r params(search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = get_milvus_client()
    instruct_query = f"Instruct: 查询相关概念\\nQuery: {query}"
    query_vec = embeddings.embed_query(instruct_query)
    logger.debug(
        "hierarchical_search embed done vec_dim=%d instruct_query=%r",
        len(query_vec), instruct_query,
    )

    hits = repo.search_sentences(client, query_vec, limit=SEARCH_LIMIT)
    logger.info("hierarchical_search milvus returned %d sentence hits", len(hits))
    if not hits:
        logger.info("hierarchical_search no hits, returning empty")
        return []

    # ---- 聚合 ----
    doc_scores: dict[int, float] = {}
    doc_titles: dict[int, str] = {}
    doc_hits: dict[int, list[SentenceHit]] = defaultdict(list)
    chap_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for h in hits:
        entity = h.get("entity", {}) or {}
        score = float(h.get("distance", 0.0))
        doc_id = entity.get("document_id")
        chap_id = entity.get("chapter_id")
        chunk_text = entity.get("chunk_text", "")
        if doc_id is None or chap_id is None:
            continue

        doc_scores[doc_id] = max(doc_scores.get(doc_id, score), score)
        doc_titles.setdefault(doc_id, entity.get("document_title", ""))
        doc_hits[doc_id].append(
            SentenceHit(chapter_id=chap_id, chunk_text=chunk_text, score=score)
        )

        chap_scores[doc_id][chap_id] = max(
            chap_scores[doc_id].get(chap_id, score), score
        )

    top_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)[:TOP_DOCS]
    logger.info(
        "hierarchical_search aggregated %d unique docs; top%d=%s",
        len(doc_scores), TOP_DOCS,
        [(d, round(doc_scores[d], 4)) for d in top_docs],
    )

    results = _aggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=doc_hits,
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_DENSE,
        with_sentence_hits=True,
        log_prefix="hierarchical_search",
    )

    total_chars = sum(r.char_count for r in results)
    logger.info(
        "hierarchical_search DONE %d results, total_chars=%d",
        len(results), total_chars,
    )
    return results


def _aggregate_to_doc_results(
    *,
    client,
    doc_scores: dict[int, float],
    doc_titles: dict[int, str],
    doc_hits: dict[int, list[SentenceHit]],
    chap_scores: dict[int, dict[int, float]],
    top_docs: list[int],
    retrieval_mode: str,
    with_sentence_hits: bool,
    log_prefix: str,
) -> list[SearchResult]:
    """共享聚合：top_docs → 每 doc 的 chapter 选择策略 → SearchResult。

    dense 与 fulltext 都复用本函数：
    - dense：sentence 召回聚合后调用，with_sentence_hits=True
    - fulltext：chapter 直接召回聚合后调用，with_sentence_hits=False

    retrieval_mode 写入 SearchResult 区分两种检索（dense=cosine 分数，fulltext=BM25 分数）。
    """
    results: list[SearchResult] = []
    for doc_id in top_docs:
        chapters = repo.query_chapters_by_document(client, doc_id)
        if not chapters:
            logger.warning(
                "%s doc_id=%d has no chapter rows, skip", log_prefix, doc_id
            )
            continue

        doc_char_count = sum(ch.get("char_count", 0) for ch in chapters)
        document_title = chapters[0].get(
            "document_title", doc_titles.get(doc_id, "")
        )

        if doc_char_count < CHAR_COUNT_THRESHOLD:
            selected = sorted(chapters, key=lambda c: c.get("chapter_id", 0))
            mode_detail = (
                f"整篇文档返回模式 (全文字数 {doc_char_count} < {CHAR_COUNT_THRESHOLD})"
            )
        else:
            scored = sorted(
                chapters,
                key=lambda c: chap_scores[doc_id].get(c.get("chapter_id"), 0.0),
                reverse=True,
            )[:TOP_CHAPTERS]
            selected = sorted(scored, key=lambda c: c.get("chapter_id", 0))
            mode_detail = (
                f"最高评分{TOP_CHAPTERS}章节返回模式 (全文字数 {doc_char_count} "
                f">= {CHAR_COUNT_THRESHOLD})"
            )
        mode = f"[{retrieval_mode}] {mode_detail}"

        sentence_hits = doc_hits.get(doc_id, []) if with_sentence_hits else []
        logger.info(
            "%s doc_id=%d title=%r chars=%d mode=%s chapters_selected=%d "
            "sentence_hits=%d",
            log_prefix, doc_id, document_title, doc_char_count, mode,
            len(selected), len(sentence_hits),
        )
        for c_idx, c in enumerate(selected):
            chap_id = int(c.get("chapter_id", 0))
            chap_title = c.get("chapter_title", "")
            chap_text = c.get("chapter_text", "")
            chap_chars = int(c.get("char_count", 0))
            preview = (chap_text or "")[:300].replace("\n", " ")
            logger.info(
                "%s doc_id=%d chapter[%d] id=%d title=%r chars=%d preview=%r",
                log_prefix, doc_id, c_idx, chap_id, chap_title,
                chap_chars, preview,
            )
            logger.debug(
                "%s doc_id=%d chapter[%d] id=%d full_text=%s",
                log_prefix, doc_id, c_idx, chap_id, chap_text,
            )

        chapter_outs = [
            ChapterOut(
                chapter_id=int(c.get("chapter_id", 0)),
                chapter_title=c.get("chapter_title", ""),
                chapter_text=c.get("chapter_text", ""),
                char_count=int(c.get("char_count", 0)),
            )
            for c in selected
        ]

        results.append(
            SearchResult(
                document_id=doc_id,
                document_title=document_title,
                document_score=doc_scores[doc_id],
                char_count=doc_char_count,
                retrieval_mode=mode,
                chapters=chapter_outs,
                sentence_hits=sentence_hits,
            )
        )
    return results


# ==========================================
# fulltext search (BM25)
# ==========================================
def fulltext_search(query: str) -> list[SearchResult]:
    """BM25 关键词检索：直接搜 chapter_collection 的 sparse_vector。

    与 hierarchical_search 的差异：
    - 不做 embedding，query 文本直传 milvus（多语言 analyzer 内部分词）
    - 召回直接是 chapter 级（无 sentence chunk 中间层）
    - sentence_hits 留空：fulltext 没有比 chapter 更细的命中粒度

    document_score 是 BM25 原始分（通常 >1，与 cosine 0-1 不可比），
    retrieval_mode='fulltext' 标记。
    """
    logger.info(
        "fulltext_search START query=%r params(bm25_search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, BM25_SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = get_milvus_client()
    analyzer_name = detect_language(query)
    logger.info("fulltext_search query_language=%s analyzer_name=%s", analyzer_name, analyzer_name)
    hits = repo.search_chapters_by_bm25(client, query, BM25_SEARCH_LIMIT, analyzer_name)
    logger.info("fulltext_search milvus returned %d chapter hits", len(hits))
    if not hits:
        logger.info("fulltext_search no hits, returning empty")
        return []

    # ---- 聚合：直接 chapter 级，无 sentence 维度 ----
    doc_scores: dict[int, float] = {}
    doc_titles: dict[int, str] = {}
    chap_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for h in hits:
        entity = h.get("entity", {}) or {}
        score = float(h.get("distance", 0.0))
        doc_id = entity.get("document_id")
        chap_id = entity.get("chapter_id")
        if doc_id is None or chap_id is None:
            continue
        doc_scores[doc_id] = max(doc_scores.get(doc_id, score), score)
        doc_titles.setdefault(doc_id, entity.get("document_title", ""))
        chap_scores[doc_id][chap_id] = max(
            chap_scores[doc_id].get(chap_id, score), score
        )

    top_docs = sorted(
        doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True
    )[:TOP_DOCS]
    logger.info(
        "fulltext_search aggregated %d unique docs; top%d=%s",
        len(doc_scores), TOP_DOCS,
        [(d, round(doc_scores[d], 4)) for d in top_docs],
    )

    results = _aggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=defaultdict(list),  # fulltext 不带 sentence 命中
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_FULLTEXT,
        with_sentence_hits=False,
        log_prefix="fulltext_search",
    )

    total_chars = sum(r.char_count for r in results)
    logger.info(
        "fulltext_search DONE %d results, total_chars=%d",
        len(results), total_chars,
    )
    return results


# ==========================================
# list / delete
# ==========================================
def list_documents() -> list[DocumentSummary]:
    """从 chapter 集合去重聚合出 document 维度。"""
    client = get_milvus_client()
    rows = repo.query_all_chapters(client)

    agg: dict[int, dict] = {}
    for r in rows:
        doc_id = r.get("document_id")
        if doc_id is None:
            continue
        agg.setdefault(
            doc_id,
            {
                "document_title": r.get("document_title", ""),
                "chapter_count": 0,
                "total_chars": 0,
            },
        )
        agg[doc_id]["chapter_count"] += 1
        agg[doc_id]["total_chars"] += int(r.get("char_count", 0))

    return [
        DocumentSummary(
            document_id=doc_id,
            document_title=data["document_title"],
            chapter_count=data["chapter_count"],
            total_chars=data["total_chars"],
        )
        for doc_id, data in sorted(
            agg.items(), key=lambda kv: kv[1]["document_title"]
        )
    ]


def delete_document(document_id: int) -> DeleteStats:
    """级联删 sentence + chapter（单文档，内部供 delete_documents 调用）。

    milvus delete by filter 不返回删除条数；这里先 query 计数再删，
    方便响应统计。两次 query 多了一点开销，但当前数据量可接受。
    """
    client = get_milvus_client()
    doc_id = int(document_id)

    chapters_before = repo.query_chapters_by_document(client, doc_id)
    deleted_chapters = len(chapters_before)

    sentences_before = client.query(
        collection_name=SENTENCE_COLL,
        filter=f"document_id == {doc_id}",
        output_fields=["document_id"],
    )
    deleted_sentences = len(sentences_before)

    repo.delete_sentences_by_document(client, doc_id)
    repo.delete_chapters_by_document(client, doc_id)

    logger.info(
        "deleted document_id=%d: chapters=%d sentences=%d",
        doc_id,
        deleted_chapters,
        deleted_sentences,
    )

    return DeleteStats(
        document_id=doc_id,
        deleted_chapters=deleted_chapters,
        deleted_sentences=deleted_sentences,
    )


def delete_documents(document_ids: list[int]) -> list[DeleteStats]:
    """批量删除文档：逐项循环，单项失败不中断后续。

    不存在的 document_id 返回 0/0 占位，前端据 deleted_chapters/sentences
    判断是否真的删到了内容。
    """
    results: list[DeleteStats] = []
    for doc_id in document_ids:
        try:
            results.append(delete_document(doc_id))
        except Exception:
            logger.exception("delete_document failed: document_id=%s", doc_id)
            results.append(
                DeleteStats(
                    document_id=int(doc_id),
                    deleted_chapters=0,
                    deleted_sentences=0,
                )
            )
    return results


# ==========================================
# collection 管理
# ==========================================
def list_collections() -> list[CollectionInfo]:
    """列出 milvus 实例上所有集合 + 行数 + 是否为 RAG 管理集合标识。

    RAG 集合优先排序（is_rag_collection=True 在前），再按字母序。
    """
    client = get_milvus_client()
    rag_set = {CHAPTER_COLL, SENTENCE_COLL}

    names = repo.list_collections(client)
    infos: list[CollectionInfo] = []
    for name in names:
        try:
            row_count = repo.get_collection_row_count(client, name)
        except Exception:
            logger.exception("get_collection_stats failed: %s", name)
            row_count = -1
        infos.append(
            CollectionInfo(
                name=name,
                row_count=row_count,
                is_rag_collection=name in rag_set,
            )
        )

    infos.sort(key=lambda c: (not c.is_rag_collection, c.name))
    return infos


def delete_collections(collection_names: list[str]) -> DeleteCollectionsResult:
    """批量删除任意集合。部分失败不阻塞其余项。

    调用前不做 ensure_collections——这是纯管理操作。
    删除后若再次调用 ingest_documents，ensure_collections 会自动重建。
    """
    client = get_milvus_client()
    deleted: list[str] = []
    failed: list[dict] = []

    for name in collection_names:
        try:
            if not repo.has_collection(client, name):
                failed.append({"name": name, "error": "collection not found"})
                continue
            repo.drop_collection(client, name)
            deleted.append(name)
            logger.info("dropped collection: %s", name)
        except Exception as e:
            logger.exception("drop_collection failed: %s", name)
            failed.append({"name": name, "error": f"{type(e).__name__}: {e}"})

    return DeleteCollectionsResult(deleted=deleted, failed=failed)


# ==========================================================
# ASYNC 实现（与上方 sync 函数一一对应）
#
# 设计原则：
# - 命名 `axxx` 与 sync 版完全对应，参数列表一致（client 类型不同）
# - 业务逻辑（分组、聚合、过滤）零改动，仅 IO 调用改 `await`
# - sync 版保留：脚本、测试、未来 CLI 工具仍可用
# - _aggregate_to_doc_results 加 _aaggregate 版本（因内部有 repo 调用）
# ==========================================================
async def aingest_documents(documents: list[DocumentInput]) -> IngestStats:
    """ingest_documents 的 async 版本。"""
    client = await get_async_milvus_client()
    await aensure_collections(client)

    grouped: dict[str, dict] = {}
    for doc in documents:
        doc_title, chap_title = _split_title(doc.title)
        document_id = _make_document_id(doc_title)
        grouped.setdefault(doc_title, {"document_id": document_id, "chapters": []})
        grouped[doc_title]["chapters"].append(
            {
                "chapter_id": int(doc.chapter_id),
                "chapter_title": chap_title,
                "paragraphs": list(doc.paragraphs),
            }
        )

    chapter_rows: list[dict] = []
    flat_meta: list[dict] = []
    for doc_title, data in grouped.items():
        document_id = data["document_id"]
        for ch in data["chapters"]:
            full_text = "\n".join(ch["paragraphs"])
            language = detect_language(full_text)
            chapter_rows.append(
                {
                    "chapter_id": ch["chapter_id"],
                    "document_id": document_id,
                    "document_title": doc_title,
                    "chapter_title": ch["chapter_title"],
                    "chapter_text": full_text,
                    "language": language,
                    "char_count": len(full_text),
                }
            )
            flat_meta.append(
                {
                    "document_id": document_id,
                    "document_title": doc_title,
                    "chapter_id": ch["chapter_id"],
                    "chapter_title": ch["chapter_title"],
                    "paragraphs": ch["paragraphs"],
                }
            )

    inserted_chapters = await repo.ainsert_chapters(client, chapter_rows)

    sentence_meta: list[dict] = []
    sentence_texts: list[str] = []
    for meta in flat_meta:
        chunks = chunk_by_sentences(
            meta["paragraphs"], window_size=CHUNK_WINDOW_SIZE, step=CHUNK_STEP
        )
        for idx, chunk_text in enumerate(chunks):
            sentence_texts.append(chunk_text)
            sentence_meta.append(
                {
                    "chunk_text": chunk_text,
                    "document_id": meta["document_id"],
                    "chapter_id": meta["chapter_id"],
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "document_title": meta["document_title"],
                    "chapter_title": meta["chapter_title"],
                }
            )

    if not sentence_texts:
        return IngestStats(inserted_chapters=inserted_chapters, inserted_sentences=0)

    logger.info("[async] embedding %d chunks...", len(sentence_texts))
    vectors = await embeddings.aembed_documents(sentence_texts)

    sentence_rows = [{**meta, "dense_vector": vec} for meta, vec in zip(sentence_meta, vectors)]
    inserted_sentences = await repo.ainsert_sentences(client, sentence_rows)

    logger.info(
        "[async] ingest done: chapters=%d sentences=%d",
        inserted_chapters, inserted_sentences,
    )
    return IngestStats(
        inserted_chapters=inserted_chapters, inserted_sentences=inserted_sentences
    )


async def ahierarchical_search(query: str) -> list[SearchResult]:
    """hierarchical_search 的 async 版本。"""
    logger.info(
        "[async] hierarchical_search START query=%r params(search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = await get_async_milvus_client()
    instruct_query = f"Instruct: 查询相关概念\\nQuery: {query}"
    query_vec = await embeddings.aembed_query(instruct_query)

    hits = await repo.asearch_sentences(client, query_vec, limit=SEARCH_LIMIT)
    logger.info("[async] hierarchical_search milvus returned %d sentence hits", len(hits))
    if not hits:
        return []

    doc_scores: dict[int, float] = {}
    doc_titles: dict[int, str] = {}
    doc_hits: dict[int, list[SentenceHit]] = defaultdict(list)
    chap_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for h in hits:
        entity = h.get("entity", {}) or {}
        score = float(h.get("distance", 0.0))
        doc_id = entity.get("document_id")
        chap_id = entity.get("chapter_id")
        chunk_text = entity.get("chunk_text", "")
        if doc_id is None or chap_id is None:
            continue
        doc_scores[doc_id] = max(doc_scores.get(doc_id, score), score)
        doc_titles.setdefault(doc_id, entity.get("document_title", ""))
        doc_hits[doc_id].append(
            SentenceHit(chapter_id=chap_id, chunk_text=chunk_text, score=score)
        )
        chap_scores[doc_id][chap_id] = max(
            chap_scores[doc_id].get(chap_id, score), score
        )

    top_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)[:TOP_DOCS]

    results = await _aaggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=doc_hits,
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_DENSE,
        with_sentence_hits=True,
        log_prefix="[async] hierarchical_search",
    )
    return results


async def _aaggregate_to_doc_results(
    *,
    client,
    doc_scores: dict[int, float],
    doc_titles: dict[int, str],
    doc_hits: dict[int, list[SentenceHit]],
    chap_scores: dict[int, dict[int, float]],
    top_docs: list[int],
    retrieval_mode: str,
    with_sentence_hits: bool,
    log_prefix: str,
) -> list[SearchResult]:
    """_aggregate_to_doc_results 的 async 版本（内部 repo 调用走 async）。"""
    results: list[SearchResult] = []
    for doc_id in top_docs:
        chapters = await repo.aquery_chapters_by_document(client, doc_id)
        if not chapters:
            logger.warning("%s doc_id=%d has no chapter rows, skip", log_prefix, doc_id)
            continue

        doc_char_count = sum(ch.get("char_count", 0) for ch in chapters)
        document_title = chapters[0].get(
            "document_title", doc_titles.get(doc_id, "")
        )

        if doc_char_count < CHAR_COUNT_THRESHOLD:
            selected = sorted(chapters, key=lambda c: c.get("chapter_id", 0))
            mode_detail = (
                f"整篇文档返回模式 (全文字数 {doc_char_count} < {CHAR_COUNT_THRESHOLD})"
            )
        else:
            scored = sorted(
                chapters,
                key=lambda c: chap_scores[doc_id].get(c.get("chapter_id"), 0.0),
                reverse=True,
            )[:TOP_CHAPTERS]
            selected = sorted(scored, key=lambda c: c.get("chapter_id", 0))
            mode_detail = (
                f"最高评分{TOP_CHAPTERS}章节返回模式 (全文字数 {doc_char_count} "
                f">= {CHAR_COUNT_THRESHOLD})"
            )
        mode = f"[{retrieval_mode}] {mode_detail}"

        sentence_hits = doc_hits.get(doc_id, []) if with_sentence_hits else []
        logger.info(
            "%s doc_id=%d title=%r chars=%d mode=%s chapters_selected=%d "
            "sentence_hits=%d",
            log_prefix, doc_id, document_title, doc_char_count, mode,
            len(selected), len(sentence_hits),
        )

        chapter_outs = [
            ChapterOut(
                chapter_id=int(c.get("chapter_id", 0)),
                chapter_title=c.get("chapter_title", ""),
                chapter_text=c.get("chapter_text", ""),
                char_count=int(c.get("char_count", 0)),
            )
            for c in selected
        ]

        results.append(
            SearchResult(
                document_id=doc_id,
                document_title=document_title,
                document_score=doc_scores[doc_id],
                char_count=doc_char_count,
                retrieval_mode=mode,
                chapters=chapter_outs,
                sentence_hits=sentence_hits,
            )
        )
    return results


async def afulltext_search(query: str) -> list[SearchResult]:
    """fulltext_search 的 async 版本（BM25 路径不调 embedding）。"""
    logger.info(
        "[async] fulltext_search START query=%r params(bm25_search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, BM25_SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = await get_async_milvus_client()
    analyzer_name = detect_language(query)
    hits = await repo.asearch_chapters_by_bm25(client, query, BM25_SEARCH_LIMIT, analyzer_name)
    logger.info("[async] fulltext_search milvus returned %d chapter hits", len(hits))
    if not hits:
        return []

    doc_scores: dict[int, float] = {}
    doc_titles: dict[int, str] = {}
    chap_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for h in hits:
        entity = h.get("entity", {}) or {}
        score = float(h.get("distance", 0.0))
        doc_id = entity.get("document_id")
        chap_id = entity.get("chapter_id")
        if doc_id is None or chap_id is None:
            continue
        doc_scores[doc_id] = max(doc_scores.get(doc_id, score), score)
        doc_titles.setdefault(doc_id, entity.get("document_title", ""))
        chap_scores[doc_id][chap_id] = max(
            chap_scores[doc_id].get(chap_id, score), score
        )

    top_docs = sorted(
        doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True
    )[:TOP_DOCS]

    results = await _aaggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=defaultdict(list),
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_FULLTEXT,
        with_sentence_hits=False,
        log_prefix="[async] fulltext_search",
    )
    return results


async def alist_documents() -> list[DocumentSummary]:
    """list_documents 的 async 版本。"""
    client = await get_async_milvus_client()
    rows = await repo.aquery_all_chapters(client)

    agg: dict[int, dict] = {}
    for r in rows:
        doc_id = r.get("document_id")
        if doc_id is None:
            continue
        agg.setdefault(
            doc_id,
            {
                "document_title": r.get("document_title", ""),
                "chapter_count": 0,
                "total_chars": 0,
            },
        )
        agg[doc_id]["chapter_count"] += 1
        agg[doc_id]["total_chars"] += int(r.get("char_count", 0))

    return [
        DocumentSummary(
            document_id=doc_id,
            document_title=data["document_title"],
            chapter_count=data["chapter_count"],
            total_chars=data["total_chars"],
        )
        for doc_id, data in sorted(
            agg.items(), key=lambda kv: kv[1]["document_title"]
        )
    ]


async def adelete_document(document_id: int) -> DeleteStats:
    """delete_document 的 async 版本。

    注意：sentence count 用 client.aquery 直接查（不经 repo），与 sync 版一致。
    """
    client = await get_async_milvus_client()
    doc_id = int(document_id)

    chapters_before = await repo.aquery_chapters_by_document(client, doc_id)
    deleted_chapters = len(chapters_before)

    sentences_before = await client.query(
        collection_name=SENTENCE_COLL,
        filter=f"document_id == {doc_id}",
        output_fields=["document_id"],
    )
    deleted_sentences = len(sentences_before)

    await repo.adelete_sentences_by_document(client, doc_id)
    await repo.adelete_chapters_by_document(client, doc_id)

    logger.info(
        "[async] deleted document_id=%d: chapters=%d sentences=%d",
        doc_id, deleted_chapters, deleted_sentences,
    )

    return DeleteStats(
        document_id=doc_id,
        deleted_chapters=deleted_chapters,
        deleted_sentences=deleted_sentences,
    )


async def adelete_documents(document_ids: list[int]) -> list[DeleteStats]:
    """delete_documents 的 async 版本；单项失败不中断。"""
    results: list[DeleteStats] = []
    for doc_id in document_ids:
        try:
            results.append(await adelete_document(doc_id))
        except Exception:
            logger.exception("[async] delete_document failed: document_id=%s", doc_id)
            results.append(
                DeleteStats(
                    document_id=int(doc_id),
                    deleted_chapters=0,
                    deleted_sentences=0,
                )
            )
    return results


async def alist_collections() -> list[CollectionInfo]:
    """list_collections 的 async 版本。"""
    client = await get_async_milvus_client()
    rag_set = {CHAPTER_COLL, SENTENCE_COLL}

    names = await repo.alist_collections(client)
    infos: list[CollectionInfo] = []
    for name in names:
        try:
            row_count = await repo.aget_collection_row_count(client, name)
        except Exception:
            logger.exception("[async] get_collection_stats failed: %s", name)
            row_count = -1
        infos.append(
            CollectionInfo(
                name=name,
                row_count=row_count,
                is_rag_collection=name in rag_set,
            )
        )

    infos.sort(key=lambda c: (not c.is_rag_collection, c.name))
    return infos


async def adelete_collections(collection_names: list[str]) -> DeleteCollectionsResult:
    """delete_collections 的 async 版本。"""
    client = await get_async_milvus_client()
    deleted: list[str] = []
    failed: list[dict] = []

    for name in collection_names:
        try:
            if not await repo.ahas_collection(client, name):
                failed.append({"name": name, "error": "collection not found"})
                continue
            await repo.adrop_collection(client, name)
            deleted.append(name)
            logger.info("[async] dropped collection: %s", name)
        except Exception as e:
            logger.exception("[async] drop_collection failed: %s", name)
            failed.append({"name": name, "error": f"{type(e).__name__}: {e}"})

    return DeleteCollectionsResult(deleted=deleted, failed=failed)
