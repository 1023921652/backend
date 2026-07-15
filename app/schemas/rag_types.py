"""RAG 接口的 Pydantic 模型。

文档 → chapter → sentence chunk 三层结构：
- DocumentInput：API 接收的最小单元（等同于 raw_documents.json 里一项）
  title 形如「大文章标题 - 章节标题」，service 层负责拆分。
- IngestStats / DeleteStats：批量操作统计
- SearchResult / ChapterOut / SentenceHit：父子查询响应
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInput(BaseModel):
    """单条 chapter 提交载荷；多次提交相同 document_title 会追加 chapter。"""
    title: str = Field(..., description='"大文章标题 - 章节标题"；无分隔符时整段视为 doc 与 chap 同名')
    chapter_id: int = Field(..., ge=0, description="文档内唯一；不同文档可重复")
    paragraphs: list[str] = Field(..., min_length=1, description="段落列表，会 join 成 chapter_text")


class IngestStats(BaseModel):
    inserted_chapters: int
    inserted_sentences: int


class SentenceHit(BaseModel):
    """sentence_collection 召回的 chunk 命中。"""
    chapter_id: int
    chunk_text: str
    score: float


class ChapterOut(BaseModel):
    chapter_id: int
    chapter_title: str
    chapter_text: str
    char_count: int


class SearchResult(BaseModel):
    """父子查询结果：聚合到 document 维度。"""
    document_id: int
    document_title: str
    document_score: float
    char_count: int
    retrieval_mode: str
    chapters: list[ChapterOut]
    sentence_hits: list[SentenceHit]


# ==========================================
# Contextual RAG 专用 DTO
# contextual_rag 子包的 sentence 集合比原版多一个 context 字段（LLM 生成的
# 上下文摘要），因此需要对应的命中/结果类型。其余 DTO（DocumentInput /
# IngestStats / SearchRequest / DocumentSummary / DeleteStats / CollectionInfo
# 等）全部复用上面的定义，不重复。
# ==========================================
class ContextualSentenceHit(BaseModel):
    """contextual_sentence_collection 召回的 chunk 命中。

    比 SentenceHit 多一个 context 字段：LLM 生成的文档级上下文摘要。
    """
    chapter_id: int
    chunk_text: str
    score: float
    context: str


class ContextualSearchResult(BaseModel):
    """Contextual RAG 父子查询结果：聚合到 document 维度。

    结构与 SearchResult 一致，仅 sentence_hits 类型换成 ContextualSentenceHit
    （带 context 字段）。
    """
    document_id: int
    document_title: str
    document_score: float
    char_count: int
    retrieval_mode: str
    chapters: list[ChapterOut]
    sentence_hits: list[ContextualSentenceHit]


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int | None = Field(None, description="预留参数，当前由配置控制")


class DocumentSummary(BaseModel):
    document_id: int
    document_title: str
    chapter_count: int
    total_chars: int


class DeleteStats(BaseModel):
    document_id: int
    deleted_chapters: int
    deleted_sentences: int


class DeleteDocumentsRequest(BaseModel):
    """批量删除文档的请求载荷。"""
    document_ids: list[int] = Field(..., min_length=1)


class CollectionInfo(BaseModel):
    """单个集合的元信息。is_rag_collection 标记是否为 RAG 管理的两个集合。"""
    name: str
    row_count: int
    is_rag_collection: bool


class DeleteCollectionsRequest(BaseModel):
    """批量删除集合的请求载荷。允许任意集合名，前端自行二次确认。"""
    collection_names: list[str] = Field(..., min_length=1)


class DeleteCollectionsResult(BaseModel):
    """批量删除集合结果：部分成功不阻塞其余项。"""
    deleted: list[str]
    failed: list[dict]
