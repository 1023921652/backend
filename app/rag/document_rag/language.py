"""插入时检测 chapter_text 语言，写入 language 字段。

language 字段值必须正好等于 MULTI_LANG_ANALYZER_PARAMS["analyzers"] 的 key
（chinese / english / default）—— milvus 的 by_field 据此选择 analyzer。
"""
from __future__ import annotations

import logging

from langdetect import DetectorFactory, detect

# langdetect 默认非确定性；固定 seed 让同一文本多次检测结果一致。
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

# langdetect ISO code -> milvus analyzer key
_LANG_MAP = {
    "zh-cn": "chinese",
    "zh-tw": "chinese",
    "en": "english",
}
# 其余语言（ja/ko/fr/de/...）→ "default"（ICU tokenizer 兜底）


def _unicode_fallback(text: str) -> str | None:
    """短文本 / langdetect 失败时走 Unicode 范围判断。

    CJK 主块命中 → chinese；ASCII 字母命中 → english；都没命中返回 None。
    """
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
    """返回 chinese / english / default 之一。

    - 长度 < 10 直接走 unicode 兜底（langdetect 短文本置信度低）
    - langdetect 抛异常（空文本/纯标点）也走 unicode 兜底
    - 都失败返回 fallback
    """
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
