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

`pyproject.toml` 中 `[tool.pytest.ini_options]` 配置了 `pythonpath = ["."]`（绝对导入）+ `asyncio_mode = "auto"`（pytest-asyncio 自动识别 `async def test_*`，无需 `@pytest.mark.asyncio` 装饰器）。测试用 `from app.main import app`，不要改成相对导入。

## 外部服务依赖

服务启动时需要以下外部进程可达（参数见 `.env`）：
- **Redis**（默认 `localhost:6379/db=0`）：LangGraph checkpointer，保存多轮会话历史。连不上会降级为无状态 agent。slowapi 多 pod 限流也复用 Redis（`RATE_LIMIT_REDIS_URI`）。
- **Milvus**（默认 `http://localhost:19530`）：向量库。不可达时 `/v1/rag/*` 接口返回 503；但 agent 启动只 import 工具函数、不立即连接，单 RAG 故障不会拖垮整个应用。

**LLM provider 通过 `LLM_PROVIDER` env 切换**（`app/agent/llm/__init__.py::_REGISTRY`）：
- `dashscope`（当前默认）→ `app/agent/llm/dashscope.py`：Qwen 多模态（DashScope OpenAI 兼容接口），支持 image_url 输入
- `deepseek` → `app/agent/llm/deepseek.py`：纯文本

新 provider 加入：建 `xxx.py` 暴露 `xxx_llm` 单例，在 `_REGISTRY` 加一行懒加载 lambda。Embedding 固定走 DashScope `text-embedding-v4`（2048 维）。

## 高层架构

整个 `app/` 是一个**把 LangChain Agent 包装成 OpenAI 标准 `/v1/chat/completions` 接口**的服务。Agent 在启动时由 `lifespan` 构建一次，注入 MCP 工具与 RAG 工具；请求期不再重建 agent，只通过 `thread_id` 从 Redis 恢复各自会话历史。

### 请求主链路

```
HTTP /v1/chat/completions   (app/api/v1/chat.py)
   ↓ get_agent(request) 拿 app.state.agent 单例
   ↓ _resolve_thread_id 解析多轮窗口标识
app/services/chat_service.py::stream_chat / nonstream_chat
   ├─ OpenWebUI task 请求（follow_ups / tags / 标题）→ 直走 LLM，不进 agent、不写 checkpoint
   └─ 普通对话 → 只取最后一条 user 消息（OpenWebUI 全量历史与 checkpoint 重复）
                 → agent.ainvoke / astream_events，config 里带 thread_id
```

关键分流逻辑：
- **task 请求识别**（`is_openwebui_task`）：最后一条消息含 `### Task:` 或 `<chat_history>` 标记 → 绕过 agent，直接 `llm.ainvoke/astream`，避免污染 Redis 会话历史。
- **只传最后一条消息**（`_take_last`）：OpenWebUI 每轮把全量历史发回，agent 的 checkpointer 已存历史，重复传入会重复计费 + 上下文污染。
- **thread_id 优先级**（`_resolve_thread_id`）：`metadata.chat_id` → `metadata.session_id` → `X-OpenWebUI-Chat-Id` header → `X-Thread-Id` header → `X-OpenWebUI-User-Id` → 首条 user 消息 sha1 → uuid 兜底。命中的 thread_id 通过响应头 `X-Thread-Id` 回传客户端。

### 启动 lifespan（`app/core/lifespan.py`）

```
tools_context()   ← app/mcp/client.py
   └─ 以 stdio 子进程启动 `python -m app.mcp.main`，加载 MCP tools，yield 期间子进程存活
append rag_tools（rag_simple_search / rag_decomposed_search / rag_fulltext_search）
set_agent(mcp_tools=mcp_tools + rag_tools)   ← app/agent/main.py
   └─ langchain.agents.create_agent(deepseek_llm, tools, CustomAgentState,
                                     OrphanToolCallSanitizerMiddleware, Redis checkpointer)
                                     → 写到 app.state.agent
```

降级链：MCP 启动失败 → 仅用 RAG 工具构建 agent；agent 构建失败 → `app.state.agent=None`，HTTP 路由返回 503；Redis 异常 → checkpointer=None 的无状态 agent。改启动顺序时务必保留这三层降级。

### 目录与职责

| 路径 | 职责 |
|---|---|
| `app/main.py` | FastAPI 入口；`load_dotenv(".env")` 最先执行；注册中间件、异常处理器、`register_rate_limiter`、`chat_router`、`rag_router`、`contextual_rag_router`。暴露 `/health`（liveness）、`/ready`（readiness）。保留 `/items/*` demo 路由供 `test_main.py` 覆盖。 |
| `app/api/v1/chat.py` | OpenAI 兼容 `/v1/chat/completions`（流式 + 非流式）与 `/v1/models`。 |
| `app/api/v1/rag.py` | RAG REST（**全 async**）：`/v1/rag/documents` 插入/列表/删除、`/v1/rag/search` 父子语义检索、`/v1/rag/search/fulltext` BM25 字面检索、`/v1/rag/collections` 管理。`ingest` / `search` 路由挂 slowapi 限流。每次调用前 `_ensure_ready()`（async）确保 Milvus 集合存在。 |
| `app/api/v1/contextual_rag.py` | Contextual RAG REST（**当前仍同步**，本期未做 async 迁移）。 |
| `app/services/chat_service.py` | OpenAI ↔ LangChain 消息映射 + 流式/非流式分流（见上文）+ `openai.APIError` 错误处理（流式下用 SSE error chunk + `[DONE]` 收尾，非流式下转 `HTTPException`）。 |
| `app/agent/` | LangChain Agent 工厂（`main.py`）、LLM provider 注册表（`llm/__init__.py::_REGISTRY`）、Redis checkpointer 工厂（`config/redis_config.py`）、`CustomAgentState`（`state/`）、孤儿 tool_calls 清洗中间件（`middleware/context.py`）。 |
| `app/mcp/` | MCP 服务端（`main.py`，stdio 启动）+ `client.py` 提供 `tools_context()` 异步上下文管理器，加载工具时挂 `_CallLogger` 拦截器在 stdout 打印每次 tool 调用与结果。 |
| `app/rag/` | `embedding.py`（Qwen text-embedding-v4 单例，原生支持 `aembed_*`）、`document_rag/`（service/repository/schemas/chunking/language/config 分层，**全 async 化**，见下）、`contextual_rag/`（仍 sync，结构与 document_rag 一致但 sentence 集合多一个 `context` 字段）、`raw_documents.json` 测试数据。 |
| `app/core/` | 跨切面：`lifespan.py`、`logging.py`（按大小切割 + 请求 ID 注入）、`middleware.py`（`X-Request-ID` + contextvars）、`errors.py`（按路径前缀分流错误格式）、`context.py`（contextvars）、`concurrency.py`（`LLM_SEMAPHORE` / `MILVUS_INSERT_SEMAPHORE` 进程级并发治理）、`rate_limit.py`（slowapi limiter + OpenAI 格式 429 handler）。 |
| `app/schemas/` | Pydantic 模型：`openai_types.py`（OpenAI 兼容请求/响应，`content` 支持 `str | list[dict]` 多模态 parts）、`rag_types.py`（RAG DTO，含 `ContextualSentenceHit` / `ContextualSearchResult`）。 |
| `app/routers/` `app/dependencies/` `app/static/` `app/tests/` | FastAPI 教程样板，仅 `tests/test_main.py` / `tests/test_chat_completions.py` / `tests/test_rag_async.py` / `tests/test_rate_limit.py` 仍被 `pytest` 收集（`X-Token: coneofsilence` 硬编码）。`dependencies.py` 的 `get_token_header` / `get_query_token` 未被 `main.py` 引用。 |

### RAG 子包（`app/rag/document_rag/` 与 `app/rag/contextual_rag/`）

两级 Milvus 集合：`chapter_collection`（父，整章节文本 + BM25 sparse_vector）与 `sentence_collection`（子，sentence chunk dense vector）。检索默认走**父子查询**：先在 sentence 层召回 top-N chunk → 聚合到 doc/chapter → 返回 top docs；另有 BM25 全文检索路径用于精确术语/型号字面匹配。`contextual_rag` sentence 集合额外多一个 `context VARCHAR(1024)` 字段（Anthropic Contextual Retrieval：每 chunk 调 LLM 生成上下文摘要）。

关键约定：
- **`document_id = CRC32(document_title)`**：跨插入稳定，同名 doc 第二次插入是追加 chapter。
- **`chapter_id` 在文档内唯一，不同文档可重复**：所有 chapter 查询必须配合 `document_id` 使用。
- **多语言分词**：`chapter_text` 写入时按 `language` 字段值选择 analyzer（chinese/english/icu 兜底），必须通过 `schemas.py` 的 `multi_analyzer_params=` 传入（不是 `analyzer_params=`），否则 pymilvus 按 1100 报错。
- **langdetect 决定 language 字段**：`by_field="language"` 的值必须正好等于 `analyzers` 的 key，`detect_language` 已保证这一点。

**异步路径（仅 document_rag）**：repository / service / endpoint / `@tool` 全 async，命名约定是 sync 版加 `a` 前缀（`insert_chapters` → `ainsert_chapters`，`ingest_documents` → `aingest_documents`）。sync 版保留供脚本/测试用。`milvus_client.py` 同时暴露 `get_milvus_client()`（sync `MilvusClient`）与 `get_async_milvus_client()`（`AsyncMilvusClient`，pymilvus 3.0+）。`@tool` 函数是 `async def`，langgraph agent 通过 `StructuredTool.coroutine` 直接 await，消除 threadpool hop。`rag_decomposed_search` 的子查询用 `asyncio.gather` 并行检索。

**contextual_rag 仍是同步**：异步化留待后续（核心收益是 `situate_context` 的 LLM 调用并行化，`asyncio.gather + Semaphore` 可加速 ingest 5-10×）。`app/core/concurrency.py` 的 `LLM_SEMAPHORE` 已就位但当前 service 层未接入。

### Agent 中间件

`OrphanToolCallSanitizerMiddleware`（`app/agent/middleware/context.py`）在每次 model 调用前剔除孤儿 tool_calls——即某条 AIMessage 的 `tool_calls` 中存在 id 没有匹配的 ToolMessage 响应（典型成因：工具改名、流式中断）。DeepSeek/OpenAI 校验失败会返回 400「An assistant message with 'tool_calls' must be followed by tool messages」。**改工具名或删工具时，这个中间件让旧 thread 历史仍可用**，不要随便移除。

## 环境与密钥约定

- 所有可配置参数集中在项目根目录 `.env`：第三方 key（DeepSeek、LangSmith、Tavily、LangSearch）、Redis 连接、Milvus 连接、embedding 配置、LLM 模型与温度、Chat API 默认模型与已知模型列表、RAG 切分/检索/索引参数、日志路径与切割阈值、`LOG_LEVEL_THREAD` 诊断级别等。
- `app/main.py` 启动早期 `load_dotenv(".env")` 加载到 `os.environ`，因此各模块直接 `os.getenv(...)` 读取即可，**不要重复 `load_dotenv`**。
- `.env` 用 python-dotenv 解析，**不要**写 `export VAR=...` 前缀，**不要**给值加引号。
- LangSmith tracing 通过 `LANGSMITH_*` 变量开启（默认 `LANGSMITH_TRACING=false`），开发 Agent 时观察链路日志走这里。

## 关键代码约定（踩过的坑）

- **`state_redis_client` 绝对不要加 `decode_responses=True`**（`app/agent/config/redis_config.py` 已注释强调）：LangGraph checkpointer 需要 bytes，加了这个参数会导致反序列化失败。
- **MCP 子进程必须在 agent 整个生命周期内存活**：`tools_context()` 用 `async with client.session(...)` 包住 `yield`，退出时自动 kill。不要把 session 的生命周期绑到单个请求。
- **pytest 绝对导入 + asyncio_mode=auto**：`pythonpath=["."]` + `asyncio_mode = "auto"` 已配。测试用 `from app.main import app`，不要改成相对导入。`async def test_*` 自动识别，无需 `@pytest.mark.asyncio`。
- **RAG 工具与 API 共享 service 层**：`app/rag/document_rag/__init__.py` 同时导出 service 函数和 LangChain tool，上层（`api/v1/rag.py` 用 service，`lifespan.py` 用 tool）都从这里取，不要绕过。
- **流式响应头**：`stream_chat` 返回的 SSE 必须带 `Cache-Control: no-cache`、`X-Accel-Buffering: no`、`Connection: keep-alive`，否则反向代理会缓冲导致流式失效。
- **流式错误处理**：`stream_chat` 已发 `200 + headers` 后无法改 status，LLM 异常（`openai.APIError`）时用 SSE error chunk（`[stream error] ...`）+ 正常 `stop` + `[DONE]` 收尾，不要让异常冒泡导致连接 RST。非流式路径则抛 `HTTPException`，由 `app/core/errors.py` 转 OpenAI `ErrorResponse`。
- **多模态 content**：`ChatMessage.content` 类型是 `Optional[str | list[dict]]`。OpenWebUI 发图时是 `list[dict]`（含 `image_url` part）。下游必须先用 `chat_service.content_to_text` 归一化（除非 user + `_content_has_multimodal=True` 时把 list 透传给 vision LLM）。DeepSeek 不支持 image_url，多模态请求必须用 dashscope provider。
- **限流装饰器签名**：slowapi 的 `@limiter.limit("N/minute")` 要求 endpoint 第一个参数是 `request: Request`，否则运行期报错。装饰器顺序：路由装饰器在外，限流在内（`@router.post(...)` → `@limiter.limit(...)` → `async def`）。
- **AsyncMilvusClient 双检锁用 `asyncio.Lock`**（不是 `threading.Lock`），且 lock 必须在 event loop 内创建（模块加载时构造 `asyncio.Lock()` 即可，uvicorn 单 worker 下 loop 与进程同生命周期）。

## 健康检查 / 部署

`/health`（liveness，不查外部依赖）与 `/ready`（readiness，仅校验 `app.state.agent` 不为 None）。**`/ready` 故意不 ping Redis/Milvus**：共享依赖 down 时所有 pod 同时 not-ready 会造成雪崩；外部依赖故障让具体请求自己报错（RAG 503、chat 流式错误 chunk）。

生产部署（k8s）：单 pod 单 uvicorn worker（多 worker 会复制 MCP 子进程内存），并发靠 `Deployment.replicas` 横向扩。`terminationGracePeriodSeconds` ≥ 60s（流式回答收尾 + MCP 子进程清理）。日志改 stdout（pod 删除后 `logs/` 目录文件丢失）。多 pod 限流必须设 `RATE_LIMIT_REDIS_URI` 切共享 backend。

## Agent 工作目录约定

Agent 的系统提示要求模型在调用 `write_file` / `ls` / `edit_file` 等工具时使用相对路径（不要以 `/` 开头、不要写物理绝对路径）。修改 Agent 相关代码或 MCP 工具时需要保留这一约定。
