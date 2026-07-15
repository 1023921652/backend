"""RAG document_rag 包入口。

对外暴露 service 层函数 + LangChain tool，让上层（api/v1/rag.py、agent）
不必关心 milvus / embedding / chunking 的细节。

注意：所有涉及 chapter_id 的查询都必须配合 document_id 使用——
chapter_id 在单文档内唯一，不同文档可重复。
"""
from app.rag.document_rag.service import (
    delete_collections,
    delete_document,
    delete_documents,
    hierarchical_search,
    ingest_documents,
    list_collections,
    list_documents,
)
from app.rag.document_rag.tools import (
    rag_decomposed_search,
    rag_fulltext_search,
    rag_simple_search,
)

__all__ = [
    "ingest_documents",
    "hierarchical_search",
    "list_documents",
    "delete_document",
    "delete_documents",
    "list_collections",
    "delete_collections",
    "rag_simple_search",
    "rag_decomposed_search",
    "rag_fulltext_search",
]
