"""Contextual RAG 配置：全部从环境变量读，集中管理。

与 document_rag/config.py 平行，独立 env 前缀 CONTEXTUAL_RAG_，
方便与原 RAG 分别调参（切分窗口、召回上限等）。

复用：MILVUS_URI / MILVUS_TOKEN / EMBEDDING_* / MULTI_LANG_ANALYZER_PARAMS
（多语言分词器配置在 document_rag.config 定义，这里直接 import 复用，
避免重复维护 analyzers 字典）。

由 app.main 启动早期 load_dotenv(".env") 加载，这里直接 os.getenv。
"""
from __future__ import annotations

import os

# ==========================================
# Milvus 连接（与 document_rag 共享）
# ==========================================
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "root:Milvus")

# ==========================================
# 集合名（独立于 document_rag）
# ==========================================
CHAPTER_COLL: str = os.getenv(
    "CONTEXTUAL_RAG_CHAPTER_COLL", "contextual_chapter_collection"
)
SENTENCE_COLL: str = os.getenv(
    "CONTEXTUAL_RAG_SENTENCE_COLL", "contextual_sentence_collection"
)

# ==========================================
# 切分参数（独立 env，方便分别调参）
# ==========================================
CHUNK_WINDOW_SIZE: int = int(os.getenv("CONTEXTUAL_RAG_CHUNK_WINDOW_SIZE", "2"))
CHUNK_STEP: int = int(os.getenv("CONTEXTUAL_RAG_CHUNK_STEP", "1"))

# ==========================================
# 检索参数
# ==========================================
SEARCH_LIMIT: int = int(os.getenv("CONTEXTUAL_RAG_SEARCH_LIMIT", "50"))
CHAR_COUNT_THRESHOLD: int = int(os.getenv("CONTEXTUAL_RAG_CHAR_COUNT_THRESHOLD", "200"))
TOP_DOCS: int = int(os.getenv("CONTEXTUAL_RAG_TOP_DOCS", "3"))
TOP_CHAPTERS: int = int(os.getenv("CONTEXTUAL_RAG_TOP_CHAPTERS", "2"))
MAX_SUBQUERIES: int = int(os.getenv("CONTEXTUAL_RAG_MAX_SUBQUERIES", "4"))

# ==========================================
# HNSW 索引超参（与 document_rag 一致即可，不暴露 env）
# ==========================================
INDEX_TYPE: str = os.getenv("RAG_INDEX_TYPE", "HNSW")
INDEX_HNSW_M: int = int(os.getenv("RAG_INDEX_HNSW_M", "16"))
INDEX_HNSW_EF: int = int(os.getenv("RAG_INDEX_HNSW_EF", "500"))

CONSISTENCY_LEVEL: str = os.getenv("RAG_CONSISTENCY_LEVEL", "Bounded")

# ==========================================
# 多语言 analyzer / 语言检测
# 复用 document_rag 的 analyzer 配置（chinese/english/default 三种，by_field="language"）
# ==========================================
DEFAULT_LANGUAGE: str = os.getenv("RAG_DEFAULT_LANGUAGE", "chinese")
SUPPORTED_LANGUAGES: tuple[str, ...] = ("chinese", "english", "default")

# 直接复用 document_rag 的多语言 analyzer 配置：避免两边配置漂移。
# chapter_text 的分词器必须与写入端 detect_language 输出一致。
from app.rag.document_rag.config import MULTI_LANG_ANALYZER_PARAMS  # noqa: E402

# ==========================================
# 检索模式标记
# ==========================================
RETRIEVAL_MODE_DENSE: str = "dense"
RETRIEVAL_MODE_FULLTEXT: str = "fulltext"

# BM25 全文检索召回上限（与 sentence SEARCH_LIMIT 解耦）
BM25_SEARCH_LIMIT: int = int(os.getenv("CONTEXTUAL_RAG_BM25_SEARCH_LIMIT", "50"))
