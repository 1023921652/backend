"""RBAC 模块配置。

通过 pydantic-settings BaseSettings 从 .env 读取。
**只服务新模块**；现有 os.getenv 调用保持原样，避免回归。
"""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # MySQL
    database_url: str = "mysql+aiomysql://root:root@localhost:3306/rbac?charset=utf8mb4"
    db_echo: bool = False

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_max_connections: int = 100
    redis_health_check_interval: int = 30
    redis_checkpoint_prefix: str = "checkpoints"
    redis_checkpoint_ttl_minutes: int = 60

    # JWT
    jwt_secret: str = "change-me-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # 环境（dev 模式下 lifespan 自动 create_all）
    env: str = "dev"


settings = Settings()
