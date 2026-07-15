"""Contextual RAG 包入口（与 document_rag 平行）。

对外暴露 service 层函数 + LangChain tool，让上层（api/v1/contextual_rag.py、
未来可能的 agent 注入）不必关心 milvus / embedding / chunking / LLM 上下文生成 的细节。

与 document_rag 的差异：ingest 时每个 chunk 会调 LLM 生成文档级上下文摘要，
拼接进 embedding 输入并独立持久化到 sentence 集合的 context 字段，提升召回精度。

注意：所有涉及 chapter_id 的查询都必须配合 document_id 使用——
chapter_id 在单文档内唯一，不同文档可重复。
"""
from app.rag.contextual_rag.service import (
    delete_collections,
    delete_document,
    delete_documents,
    fulltext_search,
    hierarchical_search,
    ingest_documents,
    list_collections,
    list_documents,
)
from app.rag.contextual_rag.tools import (
    contextual_rag_decomposed_search,
    contextual_rag_fulltext_search,
    contextual_rag_simple_search,
)

__all__ = [
    "ingest_documents",
    "hierarchical_search",
    "fulltext_search",
    "list_documents",
    "delete_document",
    "delete_documents",
    "list_collections",
    "delete_collections",
    "contextual_rag_simple_search",
    "contextual_rag_decomposed_search",
    "contextual_rag_fulltext_search",
]
