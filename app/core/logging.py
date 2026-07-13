"""结构化日志配置（标准库 logging，避免引第三方）。

特性：
- 终端 + 文件双输出
- format: [time] [LEVEL] [request_id] message
- 文件按天滚动（when=midnight），且单文件超过 100MB 也提前切割
- 保留最近 10 个备份文件
- request_id 由 RequestIdMiddleware 写入 contextvars，本模块的 filter 读出注入 LogRecord
"""
from __future__ import annotations

import logging
import logging.config
import os
from logging.handlers import TimedRotatingFileHandler

from app.core.context import request_id_ctx_var

LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")
MAX_BYTES = int(os.getenv("LOG_MAX_BYTES", str(100 * 1024 * 1024)))  # 默认 100 MB
BACKUP_COUNT = int(os.getenv("LOG_BACKUP_COUNT", "10"))
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(request_id)s] %(message)s"


class _RequestIDFilter(logging.Filter):
    """把 contextvars 里的 request_id 注入每条 LogRecord。

    contextvars 返回空字符串（启动期/后台任务/lifespan）时回退到 "-"，
    避免 format 里出现 `[]` 空字段。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        rid = request_id_ctx_var.get() or "-"
        record.request_id = rid
        return True


class _DailyAndSizeRotatingFileHandler(TimedRotatingFileHandler):
    """按天 + 按大小组合滚动的文件 handler。

    标准库 TimedRotatingFileHandler 只看时间，RotatingFileHandler 只看大小。
    本类继承前者并叠加大小判断，触发任一条件即 rollover。
    """

    def __init__(
        self,
        filename: str,
        max_bytes: int = MAX_BYTES,
        backup_count: int = BACKUP_COUNT,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        super().__init__(
            filename,
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding=encoding,
            **kwargs,
        )
        self.max_bytes = max_bytes

    def shouldRollover(self, record: logging.LogRecord) -> bool:
        # 时间触发
        if super().shouldRollover(record):
            return True
        # 大小触发：当前写入流位置 >= max_bytes
        if self.stream is None:
            self.stream = self._open()
        if self.stream.tell() >= self.max_bytes:
            return True
        return False


def _make_file_handler() -> _DailyAndSizeRotatingFileHandler:
    """dictConfig 工厂调用入口。"""
    return _DailyAndSizeRotatingFileHandler(
        LOG_FILE,
        max_bytes=MAX_BYTES,
        backup_count=BACKUP_COUNT,
    )


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": LOG_FORMAT,
        },
    },
    "filters": {
        "request_id": {
            "()": lambda: _RequestIDFilter(),
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "filters": ["request_id"],
            "stream": "ext://sys.stdout",
        },
        "file": {
            "()": _make_file_handler,
            "formatter": "default",
            "filters": ["request_id"],
        },
    },
    "loggers": {
        "app": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
        "agent": {"level": "INFO", "handlers": ["console", "file"], "propagate": False},
        # thread_id 解析路径诊断日志：默认 INFO（始终打 source/value 便于统计命中分布）
        # 诊断 header 详情时设 LOG_LEVEL_THREAD=DEBUG
        "thread_id": {
            "level": os.environ.get("LOG_LEVEL_THREAD", "INFO"),
            "handlers": ["console", "file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}


def setup_logging() -> None:
    """初始化日志配置。需要在应用启动早期调用。"""
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.config.dictConfig(LOGGING_CONFIG)
