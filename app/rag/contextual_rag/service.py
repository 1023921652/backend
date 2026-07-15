"""Contextual RAG 业务编排层。

与 document_rag/service.py 流程基本一致，区别在 ingest 阶段：
chunk 切分后、embedding 之前，对每个 chunk 调用 situate_context()
（LLM 生成文档级上下文摘要），然后把「chunk + 摘要」作为 embedding 输入。
摘要本身持久化到 sentence 集合的新 context 字段，供检索时返回。

公开函数（与 document_rag.service 同名，便于 API 层镜像）：
- ingest_documents(documents) -> IngestStats
- hierarchical_search(query) -> list[ContextualSearchResult]
- fulltext_search(query) -> list[ContextualSearchResult]
- list_documents() -> list[DocumentSummary]
- delete_document(document_id) -> DeleteStats
- delete_documents(document_ids) -> list[DeleteStats]
- list_collections() -> list[CollectionInfo]
- delete_collections(collection_names) -> DeleteCollectionsResult

document_id = CRC32(document_title)，跨插入稳定 → 同名 doc 第二次插入是追加 chapter。
chapter_id 在文档内唯一；不同文档可重复。
"""
from __future__ import annotations

import logging
import zlib
from collections import defaultdict

from app.rag.contextual_rag import repository as repo
from app.rag.contextual_rag.chunking import chunk_by_sentences
from app.rag.contextual_rag.config import (
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
from app.rag.contextual_rag.contextualizer import situate_context
from app.rag.contextual_rag.language import detect_language
from app.rag.contextual_rag.milvus_client import get_milvus_client
from app.rag.contextual_rag.schemas import ensure_collections
from app.rag.embedding import embeddings
from app.schemas.rag_types import (
    ChapterOut,
    CollectionInfo,
    ContextualSearchResult,
    ContextualSentenceHit,
    DeleteCollectionsResult,
    DeleteStats,
    DocumentInput,
    DocumentSummary,
    IngestStats,
)

logger = logging.getLogger(__name__)


def _split_title(title: str) -> tuple[str, str]:
    """title 形如「大文章标题 - 章节标题」；拆成 (document_title, chapter_title)。"""
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
    """批量插入文档（含上下文摘要生成）。

    流程：
    1. ensure_collections（首次自动建表，幂等）
    2. 按 document_title 分组、生成稳定 document_id
    3. 写 contextual_chapter_collection
    4. 切分 chunk → 每 chunk 调 LLM 生成 context → embed(chunk+context) → 写 sentence 集合

    成本：每个 chunk 一次 LLM 调用。situate_context 异常时降级为空 context，
    不阻塞写入（dense_vector 用 chunk 原文 embed）。
    """
    client = get_milvus_client()
    ensure_collections(client)

    # ---- 分组：document_title -> {document_id, chapters: [...]} ----
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
    flat_meta: list[dict] = []
    for doc_title, data in grouped.items():
        document_id = data["document_id"]
        for ch in data["chapters"]:
            full_text = "\n".join(ch["paragraphs"])
            language = detect_language(full_text)
            logger.debug(
                "[contextual_rag] ingest doc=%r chap=%r detected_language=%s",
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

    # ---- sentence 切分 + 上下文生成 + embed ----
    sentence_meta: list[dict] = []
    sentence_texts_for_embed: list[str] = []  # 喂给 embedding 的文本（chunk+context）
    for meta in flat_meta:
        chunks = chunk_by_sentences(
            meta["paragraphs"],
            window_size=CHUNK_WINDOW_SIZE,
            step=CHUNK_STEP,
        )
        # chapter 全文作为 situate_context 的 "doc" 输入
        chapter_full_text = "\n".join(meta["paragraphs"])

        for idx, chunk_text in enumerate(chunks):
            # 【关键】每个 chunk 调一次 LLM 生成上下文摘要
            context = situate_context(chapter_full_text, chunk_text)
            text_to_embed = (
                f"{chunk_text}\n\n{context}" if context else chunk_text
            )
            sentence_texts_for_embed.append(text_to_embed)
            sentence_meta.append(
                {
                    "chunk_text": chunk_text,
                    "context": context,  # 新字段，独立持久化
                    "document_id": meta["document_id"],
                    "chapter_id": meta["chapter_id"],
                    "chunk_index": idx,
                    "char_count": len(chunk_text),
                    "document_title": meta["document_title"],
                    "chapter_title": meta["chapter_title"],
                }
            )

    if not sentence_texts_for_embed:
        return IngestStats(
            inserted_chapters=inserted_chapters,
            inserted_sentences=0,
        )

    logger.info(
        "[contextual_rag] embedding %d chunks (with context)...",
        len(sentence_texts_for_embed),
    )
    vectors = embeddings.embed_documents(sentence_texts_for_embed)

    sentence_rows = []
    for meta, vec in zip(sentence_meta, vectors):
        sentence_rows.append({**meta, "dense_vector": vec})

    inserted_sentences = repo.insert_sentences(client, sentence_rows)

    logger.info(
        "[contextual_rag] ingest done: chapters=%d sentences=%d",
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
def hierarchical_search(query: str) -> list[ContextualSearchResult]:
    """父子查询（与 document_rag 一致，仅返回类型不同）。

    1. dense search sentence（chunk 级召回，含 context 字段）
    2. 按 document_id 聚合最高分；按 (document_id, chapter_id) 聚合 chapter 最高分
    3. 取 top N documents
    4. 字数 < CHAR_COUNT_THRESHOLD → 返回整 doc 的所有 chapter；否则返回 top N chapters
    """
    logger.info(
        "[contextual_rag] hierarchical_search START query=%r params(search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = get_milvus_client()
    instruct_query = f"Instruct: 查询相关概念\\nQuery: {query}"
    query_vec = embeddings.embed_query(instruct_query)

    hits = repo.search_sentences(client, query_vec, limit=SEARCH_LIMIT)
    logger.info(
        "[contextual_rag] hierarchical_search milvus returned %d sentence hits",
        len(hits),
    )
    if not hits:
        return []

    # ---- 聚合 ----
    doc_scores: dict[int, float] = {}
    doc_titles: dict[int, str] = {}
    doc_hits: dict[int, list[ContextualSentenceHit]] = defaultdict(list)
    chap_scores: dict[int, dict[int, float]] = defaultdict(dict)

    for h in hits:
        entity = h.get("entity", {}) or {}
        score = float(h.get("distance", 0.0))
        doc_id = entity.get("document_id")
        chap_id = entity.get("chapter_id")
        chunk_text = entity.get("chunk_text", "")
        context = entity.get("context", "") or ""
        if doc_id is None or chap_id is None:
            continue

        doc_scores[doc_id] = max(doc_scores.get(doc_id, score), score)
        doc_titles.setdefault(doc_id, entity.get("document_title", ""))
        doc_hits[doc_id].append(
            ContextualSentenceHit(
                chapter_id=chap_id,
                chunk_text=chunk_text,
                score=score,
                context=context,
            )
        )

        chap_scores[doc_id][chap_id] = max(
            chap_scores[doc_id].get(chap_id, score), score
        )

    top_docs = sorted(doc_scores.keys(), key=lambda d: doc_scores[d], reverse=True)[:TOP_DOCS]

    results = _aggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=doc_hits,
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_DENSE,
        with_sentence_hits=True,
        log_prefix="[contextual_rag] hierarchical_search",
    )
    return results


def _aggregate_to_doc_results(
    *,
    client,
    doc_scores: dict[int, float],
    doc_titles: dict[int, str],
    doc_hits: dict[int, list[ContextualSentenceHit]],
    chap_scores: dict[int, dict[int, float]],
    top_docs: list[int],
    retrieval_mode: str,
    with_sentence_hits: bool,
    log_prefix: str,
) -> list[ContextualSearchResult]:
    """共享聚合：top_docs → 每 doc 的 chapter 选择策略 → ContextualSearchResult。"""
    results: list[ContextualSearchResult] = []
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
            ContextualSearchResult(
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
def fulltext_search(query: str) -> list[ContextualSearchResult]:
    """BM25 关键词检索：直接搜 contextual_chapter_collection 的 sparse_vector。

    与 hierarchical_search 的差异：
    - 不做 embedding，query 文本直传 milvus（多语言 analyzer 内部分词）
    - 召回直接是 chapter 级（无 sentence chunk 中间层）
    - sentence_hits 留空：fulltext 没有比 chapter 更细的命中粒度，
      因此 ContextualSearchResult.sentence_hits 为空（context 字段也不会出现）
    """
    logger.info(
        "[contextual_rag] fulltext_search START query=%r params(bm25_search_limit=%d, top_docs=%d, "
        "char_threshold=%d, top_chapters=%d)",
        query, BM25_SEARCH_LIMIT, TOP_DOCS, CHAR_COUNT_THRESHOLD, TOP_CHAPTERS,
    )

    client = get_milvus_client()
    analyzer_name = detect_language(query)
    hits = repo.search_chapters_by_bm25(client, query, BM25_SEARCH_LIMIT, analyzer_name)
    logger.info(
        "[contextual_rag] fulltext_search milvus returned %d chapter hits",
        len(hits),
    )
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

    results = _aggregate_to_doc_results(
        client=client,
        doc_scores=doc_scores,
        doc_titles=doc_titles,
        doc_hits=defaultdict(list),
        chap_scores=chap_scores,
        top_docs=top_docs,
        retrieval_mode=RETRIEVAL_MODE_FULLTEXT,
        with_sentence_hits=False,
        log_prefix="[contextual_rag] fulltext_search",
    )
    return results


# ==========================================
# list / delete
# ==========================================
def list_documents() -> list[DocumentSummary]:
    """从 contextual chapter 集合去重聚合出 document 维度。"""
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
    """级联删 sentence + chapter（单文档）。"""
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
        "[contextual_rag] deleted document_id=%d: chapters=%d sentences=%d",
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
    """批量删除文档：逐项循环，单项失败不中断后续。"""
    results: list[DeleteStats] = []
    for doc_id in document_ids:
        try:
            results.append(delete_document(doc_id))
        except Exception:
            logger.exception(
                "[contextual_rag] delete_document failed: document_id=%s", doc_id
            )
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
    """列出 milvus 实例上所有集合 + 行数 + 是否为 contextual RAG 管理集合标识。"""
    client = get_milvus_client()
    rag_set = {CHAPTER_COLL, SENTENCE_COLL}

    names = repo.list_collections(client)
    infos: list[CollectionInfo] = []
    for name in names:
        try:
            row_count = repo.get_collection_row_count(client, name)
        except Exception:
            logger.exception(
                "[contextual_rag] get_collection_stats failed: %s", name
            )
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
    """批量删除任意集合。部分失败不阻塞其余项。"""
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
            logger.info("[contextual_rag] dropped collection: %s", name)
        except Exception as e:
            logger.exception("[contextual_rag] drop_collection failed: %s", name)
            failed.append({"name": name, "error": f"{type(e).__name__}: {e}"})

    return DeleteCollectionsResult(deleted=deleted, failed=failed)
