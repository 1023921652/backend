# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 构建与开发命令

项目使用 `uv` 管理依赖（见 `pyproject.toml` / `uv.lock`），Python 版本固定为 3.13（见 `.python-version`）。

```bash
# 安装依赖
uv sync

# 启动 FastAPI 开发服务器（[tool.fastapi] entrypoint = "app.main:app"）
uv run fastapi dev
# 或者
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 运行测试
uv run pytest

# 运行单个测试
uv run pytest app/tests/test_main.py::test_read_item
```

`pyproject.toml` 中 `[tool.pytest.ini_options].pythonpath = ["."]` 让 pytest 把项目根加入 `sys.path`，因此测试使用 `from app.main import app` 这种绝对导入，不要改成相对导入。

## 架构概览

仓库包含两个相互独立的顶层 Python 包：

### `app/` —— FastAPI HTTP 服务
- 入口：`app/main.py`（当前版本是一个内联了 `fake_db` 的最小 FastAPI 实例，路由直接写在文件里）。
- `app/main.bak.py`：被注释掉的「Bigger Applications」样板，展示了原本通过 `app.include_router(...)` 装配 `routers/items.py`、`routers/users.py` 并施加全局依赖的写法。当前 `main.py` 与 `main.bak.py` 中的路由逻辑并不一致 —— 改动路由/鉴权时务必确认实际生效的是 `main.py`。
- `app/routers/`、`app/dependencies/`、`app/static/`、`app/tests/` 遵循 FastAPI 官方教程布局；`dependencies.py` 中的 `get_token_header` / `get_query_token` 当前未被 `main.py` 引用，仅在 `main.bak.py` 中生效。
- 测试通过 `fastapi.testclient.TestClient` 调用 `app.main.app`，并依赖硬编码的 `X-Token: coneofsilence` 头。

### `agent/` —— LangChain/LangGraph 智能体
此包当前**未被 `app/main.py` 引用**，是独立的逻辑：
- `agent/main.py`：`set_agent()` 工厂函数使用 `langchain.agents.create_agent`（LangChain v1 API）构建一个有状态 Agent，传入 `deepseek_llm`、`CustomAgentState`、Redis checkpointer 以及中文系统提示。
- `agent/llm/deepseek.py`：实例化 `ChatDeepSeek`，模型名取自环境变量 `MODEL_NAME`。**关键约定**：通过 `load_dotenv("agent/.env")` 显式指定 `.env` 路径，因此无论是从哪个工作目录启动，环境变量文件都必须是 `agent/.env`。
- `agent/config/redis_config.py`：构造一个全局 redis 连接池（`max_connections=100`、`health_check_interval=30`、`socket_keepalive=True`），并通过 `get_redis_checkpointer()` 返回 `langgraph.checkpoint.redis.aio.AsyncRedisSaver`。checkpointer 配置了 `default_ttl=60` 分钟 + `refresh_on_read=True`。注意注释明确要求 `state_redis_client` **不要** 设置 `decode_responses=True`（LangGraph 需要 bytes）。
- `agent/state/main_state.py`：`CustomAgentState(AgentState)`，当前只是占位，扩展 Agent 状态时改这里。
- Redis 服务地址当前硬编码为 `localhost:6379/db=0`，本地开发需先启动 Redis。

### Agent 与 FastAPI 的衔接
`agent/main.py` 中定义了 `get_agent(request: Request)` 这个 FastAPI 依赖项占位，但尚未挂到任何路由上。若要把 Agent 接入 HTTP 层，应通过 FastAPI 依赖注入 `set_agent()` 的返回值，而不是在请求路径里直接构造。

## 环境与密钥
- 所有第三方 key（DeepSeek、LangSmith、Tavily、LangSearch 等）集中在 `agent/.env`；新增依赖外部服务的代码时遵循同一处集中管理。
- LangSmith tracing 通过 `agent/.env` 中的 `LANGSMITH_*` 变量开启，开发 Agent 时观察链路日志走这里。

## 工作目录约定
Agent 的系统提示里要求模型在调用 `write_file` / `ls` / `edit_file` 等工具时使用相对路径（不要以 `/` 开头、不要写物理绝对路径）。修改 Agent 相关代码或工具时需要保留这一约定。