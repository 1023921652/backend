"""LangChain Tool 包装：让 agent 能调用 RAG 检索。

三个工具（全部 async，由 langgraph agent 直接 await）：
- rag_simple_search：dense vector 单查询直查（语义/概念类简单问题）
- rag_decomposed_search：LLM 拆分子查询后多路 dense 并行检索再合并（复杂比较类问题）
- rag_fulltext_search：BM25 关键词检索（精确术语/型号/ID 字面匹配）

milvus / embedding / llm 调用全走 async，避免 threadpool 切换；
rag_decomposed_search 的多路子查询用 asyncio.gather 并行。
"""
from __future__ import annotations

import asyncio
import json
import logging
import re

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from app.agent.llm import get_llm
from app.rag.document_rag.config import MAX_SUBQUERIES
from app.rag.document_rag.service import afulltext_search, ahierarchical_search
from app.schemas.rag_types import SearchResult

logger = logging.getLogger(__name__)


def _format_results(results) -> str:
    """把 SearchResult list 格式化为 LLM 友好的 markdown。

    全文返回（用户决定）；agent 自行根据上下文裁剪。
    """
    if not results:
        return "（RAG 检索未命中任何文档）"

    lines: list[str] = []
    for rank, r in enumerate(results, 1):
        lines.append(f"## Rank {rank}: 文档《{r.document_title}》")
        lines.append(f"- 相似度: {r.document_score:.4f}")
        lines.append("")
        for ch in r.chapters:
            lines.append(f"### 章节《{ch.chapter_title}》")
            lines.append(ch.chapter_text)
            lines.append("")
        lines.append("---")

    return "\n".join(lines)


# ==========================================
# 简单单查询工具
# ==========================================
@tool
async def rag_simple_search(query: str) -> str:
    """检索单一对象的简单查询：直接送入向量库召回最相关文档。

    适用：单一主题的事实/概念查询（如"什么是大语言模型"、"BM25 是什么"）。
    不适用：多对象对比、多因素综合——请用 rag_decomposed_search。

    输入：自然语言查询
    输出：最相关 1-3 篇文档（含 chapter 原文，markdown 格式）

    当用户询问与知识库主题相关的问题（LLM、搜索引擎、AI、检索增强等）时调用此工具。
    无关问题（闲聊、写代码、问答计算）不要调用。
    """
    logger.info("rag_simple_search called: query=%r", query)
    try:
        results = await ahierarchical_search(query)
    except Exception as e:
        logger.exception("rag_simple_search failed")
        return f"（RAG 检索失败: {type(e).__name__}: {e}）"

    logger.info("rag_simple_search done: query=%r hits=%d", query, len(results))
    return _format_results(results)


# ==========================================
# 关键词 / 术语精确匹配工具（BM25）
# ==========================================
@tool
async def rag_fulltext_search(query: str) -> str:
    """关键词 / 术语精确匹配检索：基于 BM25 全文检索（字面命中），与 rag_simple_search（语义向量召回）互补。

    适用：
    - 精确术语、专有名词（"BM25"、"HNSW"、"ICU tokenizer"）
    - 型号 / ID / 编号（"ISO 27001"、"GPT-4"、"RFC 1918"）
    - 用户给出明确关键词、要求"包含 XX 字面"的查询

    不适用：
    - 语义/概念查询（"什么是向量检索"、"为什么需要分词"）→ 用 rag_simple_search
    - 多对象对比、复杂综合 → 用 rag_decomposed_search

    输入：关键词或术语（自然语言也可，但越精炼效果越好）
    输出：1-3 篇字面匹配度最高的文档（含 chapter 原文，markdown 格式）

    当用户问句含明确专名、型号、ID，或语义检索未命中时调用此工具。
    """
    logger.info("rag_fulltext_search called: query=%r", query)
    try:
        results = await afulltext_search(query)
    except Exception as e:
        logger.exception("rag_fulltext_search failed")
        return f"（BM25 全文检索失败: {type(e).__name__}: {e}）"

    logger.info("rag_fulltext_search done: query=%r hits=%d", query, len(results))
    return _format_results(results)


# ==========================================
# 子查询分解工具（查询分解 + 回退生成）
# ==========================================
_DECOMPOSE_PROMPT = """你是一个查询简化助手。你的任务是将用户的复杂问题转换为更适合向量检索的形式。你可以采用以下两种策略（可以同时使用，也可以根据问题特征选择其一）：

1. **查询分解**：将原问题拆分为最多 __MAX_N__ 个独立、可直接检索的子问题，每个子问题聚焦原问题的某一个具体方面（如对比题拆成各个对象的独立提问）。
2. **回退生成**：将原问题抽象为一个更通用、更直接的回溯问题（back-off question），该问题侧重获取底层事实或限制条件，以便后续检索宽泛的背景知识。

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

子问题数量不得超过 __MAX_N__ 个，且每个子问题应独立成句，避免相互依赖。

回退问题应尽可能简单、直接，聚焦于原问题背后的核心知识点（例如限制、容量、支持范围等）。

不要添加任何解释、前后缀文字或 markdown 代码块标记（只输出纯 JSON）。

示例1（分解适用，回退可选）：
输入：Milvus 和 Zilliz Cloud 在功能上有什么不同？
输出：{"sub_questions": ["Milvus 有哪些核心功能？", "Zilliz Cloud 有哪些核心功能？"], "fallback_question": ""}

示例2（回退更适用）：
输入：我有一个包含 100 亿条记录的数据集，想把它存储到 Milvus 中进行查询，可以吗？
输出：{"sub_questions": [], "fallback_question": "Milvus 可以处理的数据集大小限制是多少？"}

示例3（两种都适用）：
输入：如何在 Milvus 中使用 GPU 加速索引构建，并控制内存占用？
输出：{"sub_questions": ["Milvus 支持哪些 GPU 加速的索引类型？", "如何配置 Milvus 索引构建时的内存限制？"], "fallback_question": "Milvus 索引构建的硬件资源要求是什么？"}
"""


async def _adecompose(query: str, max_n: int) -> tuple[list[str], str]:
    """_decompose 的 async 版本；调 LLM 拆分复杂 query。

    LLM 失败 / JSON 解析失败 → 返回 ([], "")，调用方决定 fallback。
    """
    llm = get_llm()
    prompt = (
        _DECOMPOSE_PROMPT
        .replace("__MAX_N__", str(max_n))
        .replace("__QUERY__", query)
    )
    resp = await llm.ainvoke([HumanMessage(content=prompt)])
    raw = getattr(resp, "content", "") or ""

    # 兼容 LLM 偶尔输出 markdown ```json ... ``` 围栏
    m = re.search(r"\{.*}", raw, re.DOTALL)
    candidate = m.group(0) if m else raw.strip()
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        logger.warning("decompose JSON parse failed; raw=%r", raw)
        return [], ""

    if not isinstance(data, dict):
        return [], ""

    raw_subs = data.get("sub_questions", []) or []
    if not isinstance(raw_subs, list):
        raw_subs = []
    sub_qs = [str(x).strip() for x in raw_subs if str(x).strip()][:max_n]

    fallback_q = str(data.get("fallback_question", "") or "").strip()
    return sub_qs, fallback_q


def _merge_results(result_lists: list[list[SearchResult]]) -> list[SearchResult]:
    """按 document_id 合并去重；取最高 document_score；chapters/sentence_hits 合并。"""
    merged: dict[int, SearchResult] = {}
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
async def rag_decomposed_search(query: str) -> str:
    """复杂问题检索：先用 LLM 把复杂查询拆解（查询分解 + 回退生成两种策略），分别检索自建 RAG 知识库后合并去重。

    策略：
    - 查询分解：把复杂问题拆成 N 个聚焦的子问题（适合多对象对比）
    - 回退生成：构造一个更通用的回溯问题（适合容量/限制/支持范围类问题）

    适用场景：
    - 多对象对比（"A 和 B 在 X 方面有什么不同？"）
    - 多因素综合（"做 Y 需要考虑哪些方面？"）
    - 因果链条（"X 是如何影响 Y 的？"）
    - 容量/限制类（"我能否用 X 处理 Y 量数据？"）

    不适用：单一对象的简单问题（用 rag_simple_search 更快、更省 token）。

    输入：复杂的自然语言查询
    输出：合并去重后的 1-N 篇文档（含 chapter 原文，markdown 格式）
    """
    logger.info("rag_decomposed_search called: query=%r", query)

    sub_queries, fallback_q = await _adecompose(query, MAX_SUBQUERIES)

    if not sub_queries and not fallback_q:
        logger.warning("decompose failed; fallback to direct search")
        logger.info(
            "rag_decomposed_search decompose_failed fallback_direct query=%r",
            query,
        )
        try:
            return _format_results(await ahierarchical_search(query))
        except Exception as e:
            return f"（RAG 检索失败: {type(e).__name__}: {e}）"

    logger.info(
        "rag_decomposed_search decompose sub_queries=%s fallback_question=%r",
        sub_queries, fallback_q,
    )

    # 子问题 + 回退问题并行检索（asyncio.gather），不再串行
    queries_with_tag: list[tuple[str, str]] = [
        (sq, "sub_query") for sq in sub_queries
    ]
    if fallback_q:
        queries_with_tag.append((fallback_q, "fallback"))

    async def _safe_search(q: str, tag: str) -> list[SearchResult]:
        try:
            rs = await ahierarchical_search(q)
            logger.info("rag_decomposed_search %s=%r hits=%d", tag, q, len(rs))
            return rs
        except Exception:
            logger.exception("%s search failed: %s", tag, q)
            return []

    batch = await asyncio.gather(
        *[_safe_search(q, tag) for q, tag in queries_with_tag]
    )
    all_results = [r for r in batch if r]

    if not all_results:
        return "（RAG 子查询与回退问题全部失败，无结果）"

    merged = _merge_results(all_results)
    return _format_results(merged)
