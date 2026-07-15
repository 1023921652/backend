"""上下文摘要生成器：Anthropic Contextual Retrieval 的 LLM 调用封装。

对每个 chunk，让 LLM 生成一段「这个 chunk 在整篇文档中的位置/作用」摘要，
拼接在 chunk 文本后面再做 embedding，提升语义召回精度。

与 Anthropic 原版的差异：
- 不使用 anthropic SDK，改走 app.agent.llm.get_llm()（默认 DeepSeek），
  与项目其他 LLM 调用统一管理、共用 .env 配置。
- 不显式传 prompt caching 参数；DeepSeek 服务端对相同前缀自动隐式缓存，
  doc_text 放在 prompt 开头仍能受益。
- LLM 异常时返回空串降级，避免单 chunk 失败阻塞整批 ingest。
"""
from __future__ import annotations

import logging

from langchain_core.messages import HumanMessage

from app.agent.llm import get_llm

logger = logging.getLogger(__name__)


_DOCUMENT_TMPL = "<document>\n{doc_content}\n</document>"

_CHUNK_TMPL = """Here is the chunk we want to situate within the whole document:
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else."""


def situate_context(doc_text: str, chunk_text: str) -> str:
    """让 LLM 给 chunk 生成一段文档级上下文摘要。

    参数：
        doc_text: 整篇文档（或当前 chapter）的完整文本，作为上下文背景
        chunk_text: 当前 chunk 的文本

    返回：
        上下文摘要字符串。LLM 失败时返回空串（降级，不阻塞写入）。

    每个 chunk 调用一次 LLM；成本与 chunk 数量线性相关。DeepSeek 隐式
    prompt caching 对同一 doc_text 的多次调用有缓存效果。
    """
    try:
        prompt = (
            _DOCUMENT_TMPL.format(doc_content=doc_text)
            + "\n"
            + _CHUNK_TMPL.format(chunk_content=chunk_text)
        )
        resp = get_llm().invoke([HumanMessage(content=prompt)])
        context = (getattr(resp, "content", "") or "").strip()
        logger.debug(
            "situate_context ok chunk_len=%d context_len=%d",
            len(chunk_text), len(context),
        )
        return context
    except Exception:
        logger.exception(
            "situate_context failed; falling back to empty context (chunk_len=%d)",
            len(chunk_text),
        )
        return ""
