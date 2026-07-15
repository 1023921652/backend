"""RAG 配置：全部从环境变量读，集中管理。

由 app.main 启动早期 load_dotenv(".env") 加载，这里直接 os.getenv。
"""
from __future__ import annotations

import os

# Milvus 连接
MILVUS_URI: str = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "root:Milvus")

# 集合名
CHAPTER_COLL: str = os.getenv("RAG_CHAPTER_COLL", "chapter_collection")
SENTENCE_COLL: str = os.getenv("RAG_SENTENCE_COLL", "sentence_collection")

# 切分参数
CHUNK_WINDOW_SIZE: int = int(os.getenv("RAG_CHUNK_WINDOW_SIZE", "3"))
CHUNK_STEP: int = int(os.getenv("RAG_CHUNK_STEP", "3"))

# 检索参数
SEARCH_LIMIT: int = int(os.getenv("RAG_SEARCH_LIMIT", "50"))
CHAR_COUNT_THRESHOLD: int = int(os.getenv("RAG_CHAR_COUNT_THRESHOLD", "10000"))
TOP_DOCS: int = int(os.getenv("RAG_TOP_DOCS", "3"))
TOP_CHAPTERS: int = int(os.getenv("RAG_TOP_CHAPTERS", "2"))
MAX_SUBQUERIES: int = int(os.getenv("RAG_MAX_SUBQUERIES", "4"))

# HNSW 索引超参
INDEX_TYPE: str = os.getenv("RAG_INDEX_TYPE", "HNSW")
INDEX_HNSW_M: int = int(os.getenv("RAG_INDEX_HNSW_M", "16"))
INDEX_HNSW_EF: int = int(os.getenv("RAG_INDEX_HNSW_EF", "500"))

# 一致性级别：Strong / Bounded / Eventually / Session / Bounded
CONSISTENCY_LEVEL: str = os.getenv("RAG_CONSISTENCY_LEVEL", "Bounded")

# ==========================================
# 多语言 analyzer / 语言检测
# ==========================================
DEFAULT_LANGUAGE: str = os.getenv("RAG_DEFAULT_LANGUAGE", "chinese")
SUPPORTED_LANGUAGES: tuple[str, ...] = ("chinese", "english", "default")

# chapter_text 的多语言分词配置：写入时按 language 字段值选择 analyzer。
# by_field 的值必须正好等于 analyzers 的 key（detect_language 已保证）。
# 必须用 schemas.py 的 multi_analyzer_params= 传入（不是 analyzer_params=），
# 否则 pymilvus 按普通 analyzer 校验顶层字段会报 1100。
# default 用 {"tokenizer": "icu"}：ICU 对未匹配语言做通用兜底分词（官网推荐）。
MULTI_LANG_ANALYZER_PARAMS: dict = {
    "analyzers": {
        "english": {"type": "english"},
        "chinese": {"type": "chinese"},
        "default": {"tokenizer": "icu"},
    },
    "by_field": "language",
}

# ==========================================
# 检索模式标记（写入 SearchResult.retrieval_mode）
# ==========================================
RETRIEVAL_MODE_DENSE: str = "dense"
RETRIEVAL_MODE_FULLTEXT: str = "fulltext"

# BM25 全文检索召回上限（直接搜 chapter，与 sentence 的 SEARCH_LIMIT 解耦）
BM25_SEARCH_LIMIT: int = int(os.getenv("RAG_BM25_SEARCH_LIMIT", "50"))
