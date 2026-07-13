"""Embedding 服务（Qwen text-embedding-v4 via DashScope compatible API）。

embeddings 实例在模块加载时构造；OpenAIEmbeddings 是懒调用，
真正请求 API 在 embed_query/embed_documents 第一次执行时发生。
"""
import os

from langchain_openai import OpenAIEmbeddings

API_KEY = os.getenv("EMBEDDING_API_KEY", "")
BASE_URL = os.getenv(
    "EMBEDDING_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
VECTOR_DIM = int(os.getenv("EMBEDDING_VECTOR_DIM", "2048"))

embeddings = OpenAIEmbeddings(
    api_key=API_KEY,
    base_url=BASE_URL,
    model=MODEL_NAME,
    check_embedding_ctx_length=False,
    dimensions=VECTOR_DIM,
    chunk_size=10,
)
