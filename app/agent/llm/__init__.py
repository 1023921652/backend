"""LLM 注册表：通过 LLM_PROVIDER 环境变量选择具体 provider。

新增 provider 步骤：
1. 在本目录新建 xxx.py，按需读 xxx 相关 env，暴露 `xxx_llm` 单例
2. 在 _REGISTRY 注册一个懒加载 lambda（避免未使用的 provider 也被实例化）

注意：lru_cache 保证 get_default_llm 进程级只构造一次。
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Callable

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

# provider 名 -> 构造函数（懒加载：未选中的 provider 不会被 import / 实例化）
_REGISTRY: dict[str, Callable[[], BaseChatModel]] = {
    "deepseek": lambda: _import_attr("app.agent.llm.deepseek", "deepseek_llm"),
    "dashscope": lambda: _import_attr("app.agent.llm.dashscope", "dashscope_llm"),
}


def _import_attr(module: str, attr: str):
    """延迟 import；避免顶层一次性 import 所有 provider 触发不必要的副作用。"""
    import importlib

    return getattr(importlib.import_module(module), attr)


def list_providers() -> list[str]:
    """当前已注册的 provider 名列表（供调试 / Swagger 元信息用）。"""
    return list(_REGISTRY.keys())

# 新增 provider 流程（未来加 LLM 时）：
#   1. 在 app/agent/llm/ 新建 xxx.py，从 env 读配置并暴露 xxx_llm 单例
#   2. 在 __init__.py 的 _REGISTRY 加一行：
#   "xxx": lambda: _import_attr("app.agent.llm.xxx", "xxx_llm"),
#   3. 改 .env 的 LLM_PROVIDER=xxx 即切换
@lru_cache(maxsize=1)
def get_default_llm() -> BaseChatModel:
    """按 LLM_PROVIDER 环境变量返回对应 LLM 单例；进程级缓存。

    未知值回退到 'deepseek' 并告警，避免启动失败。
    """
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in _REGISTRY:
        logger.warning(
            "unknown LLM_PROVIDER=%r; available=%s; fallback to 'deepseek'",
            provider,
            list_providers(),
        )
        provider = "deepseek"
    logger.info("activating LLM provider: %s", provider)
    return _REGISTRY[provider]()


def register_provider(name: str, factory: Callable[[], BaseChatModel]) -> None:
    """运行期注册新 provider（供插件式扩展）。"""
    _REGISTRY[name.strip().lower()] = factory

def get_llm():
    """返回配置的 LLM 单例，用于 OpenWebUI task 请求的直 LLM 路径。

    具体使用哪个 provider 由 .env 的 LLM_PROVIDER 决定（默认 deepseek）。
    通过 Depends 注入便于测试替换；task 请求不走 agent、不写 checkpoint。
    """
    return get_default_llm()