"""句群切分：把段落列表按滑动窗口合成 child chunks。

每个 chunk 包含 window_size 个段落，窗口每次前进 step 段。
当 window_size == step 时窗口不重叠；< 时重叠；> 时跳过部分段落。

与 app.rag.document_rag.chunking 内容一致；为保持 contextual_rag 自包含而复制。
"""
from __future__ import annotations


def chunk_by_sentences(
    paragraphs: list[str],
    window_size: int = 3,
    step: int = 3,
) -> list[str]:
    """把段落列表按窗口合成 chunk 文本。

    - 段落末尾若已带中文标点（。！？'）则直接拼接，否则补"。"
    - 窗口滑出末尾即停止（最后一段如果切不齐完整 window 就停）
    """
    chunks: list[str] = []
    n = len(paragraphs)
    if n == 0:
        return chunks

    for i in range(0, n, step):
        window = paragraphs[i : i + window_size]
        chunk_text = "".join(
            s if s.endswith(("。", "！", "？", "‘", "’")) else s + "。"
            for s in window
        )
        chunks.append(chunk_text)
        if i + window_size >= n:
            break

    return chunks
