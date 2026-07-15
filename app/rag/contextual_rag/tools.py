"""LangChain Tool 包装：contextual RAG 检索工具（3 个）。

与 document_rag.tools 结构一致，区别：
- 函数名加 contextual_ 前缀，避免与 document_rag 工具重名
- _format_results 输出 markdown 时额外展示 context 字段

注意：本期这 3 个工具**不注入 agent**（仅 REST 暴露）。
保留实现便于未来切换策略（只需改 lifespan.py + 系统提示）。
"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.agent.llm import get_llm
from app.rag.contextual_rag.config import MAX_SUBQUERIES
from app.rag.contextual_rag.service import fulltext_search, hierarchical_search
from app.schemas.rag_types import ContextualSearchResult

logger = logging.getLogger(__name__)


def _format_results(results: list[ContextualSearchResult]) -> str:
    """把 ContextualSearchResult list 格式化为 LLM 友好的 markdown。

    与 document_rag 版本差异：sentence_hits 里每个 chunk 后展示其 context。
    """
    if not results:
        return "（Contextual RAG 检索未命中任何文档）"

    lines: list[str] = []
    for rank, r in enumerate(results, 1):
        lines.append(f"## Rank {rank}: 文档《{r.document_title}》")
        lines.append(f"- 相似度: {r.document_score:.4f}")
        lines.append("")

        for ch in r.chapters:
            lines.append(f"### 章节《{ch.chapter_title}》")
            lines.append(ch.chapter_text)
            lines.append("")

        if r.sentence_hits:
            lines.append("### 命中的 sentence chunks（含上下文摘要）：")
            for h in r.sentence_hits:
                lines.append(f"- chunk: {h.chunk_text}")
                if h.context:
                    lines.append(f"  context: {h.context}")
            lines.append("")

        lines.append("---")

    return "\n".join(lines)


# ==========================================
# 简单单查询工具
# ==========================================
@tool
def contextual_rag_simple_search(query: str) -> str:
    """Contextual RAG 的简单单查询：用带上下文摘要的 dense 向量召回。

    与 rag_simple_search 的差异：sentence chunk 在 embedding 时附带了
    LLM 生成的文档级上下文摘要，对孤立 chunk（缺主语/上下文）的召回更准。

    适用：单一主题的事实/概念查询。
    输入：自然语言查询
    输出：最相关文档（含 chapter 原文 + chunk 的 context 摘要）
    """
    logger.info("[contextual_rag] contextual_rag_simple_search called: query=%r", query)
    try:
        results = hierarchical_search(query)
    except Exception as e:
        logger.exception("[contextual_rag] contextual_rag_simple_search failed")
        return f"（Contextual RAG 检索失败: {type(e).__name__}: {e}）"

    logger.info(
        "[contextual_rag] contextual_rag_simple_search done: query=%r hits=%d",
        query, len(results),
    )
    return _format_results(results)


# ==========================================
# 关键词 / 术语精确匹配工具（BM25）
# ==========================================
@tool
def contextual_rag_fulltext_search(query: str) -> str:
    """Contextual RAG 的 BM25 全文检索。

    与 rag_fulltext_search 行为一致：直接搜 contextual_chapter_collection
    的 sparse_vector。注意 fulltext 不返回 chunk 级命中，因此看不到 context 字段。

    适用：精确术语、型号、ID 字面匹配。
    """
    logger.info("[contextual_rag] contextual_rag_fulltext_search called: query=%r", query)
    try:
        results = fulltext_search(query)
    except Exception as e:
        logger.exception("[contextual_rag] contextual_rag_fulltext_search failed")
        return f"（Contextual BM25 检索失败: {type(e).__name__}: {e}）"

    logger.info(
        "[contextual_rag] contextual_rag_fulltext_search done: query=%r hits=%d",
        query, len(results),
    )
    return _format_results(results)


# ==========================================
# 子查询分解工具
# ==========================================
_DECOMPOSE_PROMPT = """你是一个查询简化助手。你的任务是将用户的复杂问题转换为更适合向量检索的形式。你可以采用以下两种策略（可以同时使用，也可以根据问题特征选择其一）：

1. **查询分解**：将原问题拆分为最多 __MAX_N__ 个独立、可直接检索的子问题。
2. **回退生成**：将原问题抽象为一个更通用、更直接的回溯问题。

输入复杂问题：__QUERY__

输出要求：
- 你**必须**返回一个合法的 JSON 对象，格式如下：
  ```json
  {
    "sub_questions": ["子问题1", "子问题2", "..."],
    "fallback_question": "回退问题字符串"
  }
  ```
- 如果某策略不适用，对应字段设为空数组（[]）或空字符串（""），但字段必须存在。

不要添加任何解释、前后缀文字或 markdown 代码块标记（只输出纯 JSON）。
"""


def _decompose(query: str, max_n: int) -> tuple[list[str], str]:
    """调 LLM 把复杂 query 拆成 (sub_questions, fallback_question)。"""
    llm = get_llm()
    prompt = (
        _DECOMPOSE_PROMPT
        .replace("__MAX_N__", str(max_n))
        .replace("__QUERY__", query)
    )
    resp = llm.invoke([HumanMessage(content=prompt)])
    raw = getattr(resp, "content", "") or ""

    m = re.search(r"\{.*}", raw, re.DOTALL)
    candidate = m.group(0) if m else raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("[contextual_rag] decompose JSON parse failed; raw=%r", raw)
        return [], ""

    if not isinstance(data, dict):
        return [], ""

    raw_subs = data.get("sub_questions", []) or []
    if not isinstance(raw_subs, list):
        raw_subs = []
    sub_qs = [str(x).strip() for x in raw_subs if str(x).strip()][:max_n]

    fallback_q = str(data.get("fallback_question", "") or "").strip()
    return sub_qs, fallback_q


def _merge_results(
    result_lists: list[list[ContextualSearchResult]],
) -> list[ContextualSearchResult]:
    """按 document_id 合并去重；取最高 document_score；chapters/sentence_hits 合并。"""
    merged: dict[int, ContextualSearchResult] = {}
    for results in result_lists:
        for r in results:
            existing = merged.get(r.document_id)
            if existing is None:
                merged[r.document_id] = r.model_copy(deep=True)
                continue
            if r.document_score > existing.document_score:
                existing.document_score = r.document_score
            seen_chap = {c.chapter_id for c in existing.chapters}
            for c in r.chapters:
                if c.chapter_id not in seen_chap:
                    existing.chapters.append(c)
                    seen_chap.add(c.chapter_id)
            seen_text = {h.chunk_text for h in existing.sentence_hits}
            for h in r.sentence_hits:
                if h.chunk_text not in seen_text:
                    existing.sentence_hits.append(h)
                    seen_text.add(h.chunk_text)

    return sorted(
        merged.values(), key=lambda r: r.document_score, reverse=True
    )


@tool
def contextual_rag_decomposed_search(query: str) -> str:
    """Contextual RAG 的复杂问题检索：LLM 拆解查询 → 多路 contextual 检索 → 合并。

    与 rag_decomposed_search 流程一致，但底层调 contextual 版本的
    hierarchical_search（每个 chunk 的向量都带上下文摘要）。
    """
    logger.info("[contextual_rag] contextual_rag_decomposed_search called: query=%r", query)

    sub_queries, fallback_q = _decompose(query, MAX_SUBQUERIES)

    if not sub_queries and not fallback_q:
        logger.warning("[contextual_rag] decompose failed; fallback to direct search")
        try:
            return _format_results(hierarchical_search(query))
        except Exception as e:
            return f"（Contextual RAG 检索失败: {type(e).__name__}: {e}）"

    queries_with_tag: list[tuple[str, str]] = [
        (sq, "sub_query") for sq in sub_queries
    ]
    if fallback_q:
        queries_with_tag.append((fallback_q, "fallback"))

    all_results: list[list[ContextualSearchResult]] = []
    for q, tag in queries_with_tag:
        try:
            rs = hierarchical_search(q)
            all_results.append(rs)
            logger.info(
                "[contextual_rag] decomposed %s=%r hits=%d", tag, q, len(rs)
            )
        except Exception:
            logger.exception("[contextual_rag] %s search failed: %s", tag, q)

    if not all_results:
        return "（Contextual RAG 子查询与回退问题全部失败，无结果）"

    merged = _merge_results(all_results)
    return _format_results(merged)
