"""插入时检测 chapter_text 语言，写入 language 字段。

language 字段值必须正好等于 MULTI_LANG_ANALYZER_PARAMS["analyzers"] 的 key
（chinese / english / default）—— milvus 的 by_field 据此选择 analyzer。

与 app.rag.document_rag.language 内容一致；为保持 contextual_rag 自包含而复制。
"""
from __future__ import annotations

import logging

from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

_LANG_MAP = {
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "en": "english",
}


def _unicode_fallback(text: str) -> str | None:
    if not text:
        return None
    has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    if has_cjk:
        return "chinese"
    has_latin = any("a" <= ch.lower() <= "z" for ch in text)
    if has_latin:
        return "english"
    return None


def detect_language(text: str, fallback: str = "chinese") -> str:
    """返回 chinese / english / default 之一。"""
    if not text or len(text) < 10:
        return _unicode_fallback(text) or fallback
    try:
        code = detect(text)
    except Exception:
        logger.debug("langdetect failed; fallback to unicode", exc_info=True)
        return _unicode_fallback(text) or fallback
    lang = _LANG_MAP.get(code, "default")
    logger.debug("detect_language code=%r -> %r (text_len=%d)", code, lang, len(text))
    return lang
