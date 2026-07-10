"""对 /v1/chat/completions 的非流式/流式/OpenWebUI task 分流做 mock 测试。

通过 app.dependency_overrides 同时替换 get_agent 与 get_llm，
完全隔离 Redis 和 DeepSeek API。
"""
import json
import logging

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.v1.chat import get_agent, get_llm
from app.main import app


# ==========================================
# Mock 对象
# ==========================================
class MockChunk:
    """模拟 langchain AIMessage 的 stream chunk（仅需 content 属性）。"""
    def __init__(self, content: str):
        self.content = content


class MockAgent:
    """模拟 LangChain v1 create_agent 返回的 CompiledStateGraph。
    记录 ainvoke / astream_events 调用次数与最后传入的 config，便于验证
    task 请求是否绕过 agent、以及 thread_id 是否正确透传。
    """
    def __init__(self):
        self.ainvoke_calls = 0
        self.last_messages = None
        self.last_config = None
        self.astream_calls = 0

    async def ainvoke(self, messages, config):
        self.ainvoke_calls += 1
        self.last_messages = messages
        self.last_config = config
        return {"messages": [AIMessage(content="agent reply")]}

    async def astream_events(self, messages, config, version="v2"):
        self.astream_calls += 1
        self.last_messages = messages
        self.last_config = config
        for tok in ["Hello", " world"]:
            yield {"event": "on_chat_model_stream", "data": {"chunk": MockChunk(tok)}}
        yield {"event": "on_chain_end", "data": {}}
        yield {"event": "on_chat_model_stream", "data": {"chunk": MockChunk("")}}


class MockLLM:
    """模拟 ChatDeepSeek，记录调用次数。"""
    def __init__(self):
        self.ainvoke_calls = 0
        self.astream_calls = 0
        self.last_messages = None

    async def ainvoke(self, messages):
        self.ainvoke_calls += 1
        self.last_messages = messages
        return AIMessage(content='{"follow_ups": ["q1?", "q2?"]}')

    async def astream(self, messages):
        self.astream_calls += 1
        self.last_messages = messages
        for tok in ["{", '"tags"', ":", "[]", "}"]:
            yield MockChunk(tok)


def _install_mocks(agent: MockAgent | None = None, llm: MockLLM | None = None):
    """同时 override get_agent 和 get_llm，返回传入实例（便于断言）。"""
    a = agent or MockAgent()
    l = llm or MockLLM()
    app.dependency_overrides[get_agent] = lambda: a
    app.dependency_overrides[get_llm] = lambda: l
    return a, l


# ==========================================
# 普通对话路径
# ==========================================
def test_nonstream_returns_openai_format():
    agent, llm = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert body["object"] == "chat.completion"
            assert body["model"] == "deepseek-chat"
            assert body["id"].startswith("chatcmpl-")
            assert body["choices"][0]["message"]["content"] == "agent reply"
            assert body["choices"][0]["finish_reason"] == "stop"
            # 普通对话走 agent，不走 llm
            assert agent.ainvoke_calls == 1
            assert llm.ainvoke_calls == 0
    finally:
        app.dependency_overrides.clear()


# ==========================================
# thread_id 解析（OpenWebUI 适配）
# ==========================================
def test_thread_id_from_metadata_chat_id():
    """OpenWebUI 在 metadata.chat_id 传聊天窗口 id，应作为 thread_id。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "win-xyz"},
                },
            )
            assert r.status_code == 200
            # 响应头回显实际使用的 thread_id
            assert r.headers.get("X-Thread-Id") == "win-xyz"
            # 透传给 agent 的 config.configurable.thread_id 一致
            assert agent.last_config["configurable"]["thread_id"] == "win-xyz"
            # 响应体不含 user 字段（OpenAI 标准 schema）
            assert "user" not in r.json()
    finally:
        app.dependency_overrides.clear()


def test_thread_id_falls_back_to_session_id():
    """无 chat_id 时用 metadata.session_id。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"session_id": "sess-001"},
                },
            )
            assert r.status_code == 200
            assert r.headers.get("X-Thread-Id") == "sess-001"
            assert agent.last_config["configurable"]["thread_id"] == "sess-001"
    finally:
        app.dependency_overrides.clear()


def test_thread_id_from_header():
    """无 metadata 时回退到 header X-Thread-Id。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-Thread-Id": "header-thread"},
            )
            assert r.status_code == 200
            assert r.headers.get("X-Thread-Id") == "header-thread"
            assert agent.last_config["configurable"]["thread_id"] == "header-thread"
    finally:
        app.dependency_overrides.clear()


def test_thread_id_uuid_fallback_when_no_user_message():
    """messages 里没有 user 消息时（content-hash 也为空），最终落到 uuid 兜底。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "system", "content": "you are helpful"}]},
            )
            assert r.status_code == 200
            tid = r.headers.get("X-Thread-Id")
            assert tid and len(tid) >= 8
            # uuid 不含 'fp-' 前缀
            assert not tid.startswith("fp-")
            assert agent.last_config["configurable"]["thread_id"] == tid
    finally:
        app.dependency_overrides.clear()


def test_thread_id_from_content_hash():
    """无任何外部标识、messages 含 user 消息时，走首条 user 消息内容哈希兜底。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "你好，我是张三"}]},
            )
            assert r.status_code == 200
            tid = r.headers.get("X-Thread-Id")
            assert tid and tid.startswith("fp-")
            assert agent.last_config["configurable"]["thread_id"] == tid
    finally:
        app.dependency_overrides.clear()


def test_content_hash_stable_across_turns():
    """同一窗口不同轮次（首条 user 消息相同）应解析出同一 thread_id。"""
    _install_mocks()
    try:
        with TestClient(app) as client:
            # 第 1 轮：单条 user
            r1 = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "你好"}]},
            )
            # 第 2 轮：OpenWebUI 全量历史（首条 user 不变）
            r2 = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [
                          {"role": "user", "content": "你好"},
                          {"role": "assistant", "content": "你好！有什么可以帮你？"},
                          {"role": "user", "content": "今天天气怎么样"},
                      ]},
            )
            assert r1.status_code == 200 and r2.status_code == 200
            assert r1.headers.get("X-Thread-Id") == r2.headers.get("X-Thread-Id")
    finally:
        app.dependency_overrides.clear()


def test_content_hash_normalizes_whitespace_and_case():
    """首条 user 消息的大小写/前后空白差异应归一化到同一指纹。"""
    _install_mocks()
    try:
        with TestClient(app) as client:
            r1 = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "  Hello World  "}]},
            )
            r2 = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "hello world"}]},
            )
            assert r1.headers.get("X-Thread-Id") == r2.headers.get("X-Thread-Id")
    finally:
        app.dependency_overrides.clear()


def test_thread_id_from_openwebui_user_header():
    """仅 header X-OpenWebUI-User-Id 时走用户级弱标识（无窗口级标识的兜底）。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "hi"}]},
                headers={"X-OpenWebUI-User-Id": "user-007"},
            )
            assert r.status_code == 200
            assert r.headers.get("X-Thread-Id") == "user-007"
            assert agent.last_config["configurable"]["thread_id"] == "user-007"
    finally:
        app.dependency_overrides.clear()


def test_response_body_has_no_user_field():
    """OpenAI 标准 ChatCompletionResponse 不含 user 字段。"""
    _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "hi"}]},
            )
            body = r.json()
            assert "user" not in body
    finally:
        app.dependency_overrides.clear()


def test_only_last_message_is_passed_to_agent():
    """多消息请求时，只把最后一条 user 消息传给 agent（不与 checkpoint 历史重复）。"""
    agent, _ = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "user", "content": "earlier"},
                        {"role": "assistant", "content": "old reply"},
                        {"role": "user", "content": "latest question"},
                    ],
                },
            )
            assert r.status_code == 200
            # agent.ainvoke 收到 {"messages": [...]}，只应包含最后一条 user 消息
            assert agent.last_messages is not None
            received = agent.last_messages["messages"]
            assert len(received) == 1
            assert received[0].content == "latest question"
    finally:
        app.dependency_overrides.clear()


def test_stream_returns_sse_with_done_marker():
    agent, llm = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": True,
                },
            )
            assert r.status_code == 200
            assert "text/event-stream" in r.headers["content-type"]
            raw_lines = [ln.rstrip("\r") for ln in r.text.split("\n")]
            datas = [ln[6:] for ln in raw_lines if ln.startswith("data: ")]
            assert datas[-1] == "[DONE]"

            first = json.loads(datas[0])
            assert first["object"] == "chat.completion.chunk"
            assert first["choices"][0]["delta"]["role"] == "assistant"
            assert first["choices"][0]["delta"]["content"] == "Hello"

            second = json.loads(datas[1])
            assert "role" not in second["choices"][0]["delta"]
            assert second["choices"][0]["delta"]["content"] == " world"

            last_chunk = json.loads(datas[-2])
            assert last_chunk["choices"][0]["finish_reason"] == "stop"

            ids = {json.loads(d)["id"] for d in datas if d != "[DONE]"}
            assert len(ids) == 1

            # 普通对话流式走 agent，不走 llm
            assert agent.astream_calls == 1
            assert llm.astream_calls == 0
    finally:
        app.dependency_overrides.clear()


# ==========================================
# OpenWebUI task 分流路径
# ==========================================
_OPENWEBUI_FOLLOWUPS_PROMPT = (
    "### Task:\nSuggest 3-5 relevant follow-up questions...\n"
    "### Output:\nJSON format: { \"follow_ups\": [...] }\n"
    "### Chat History:\n<chat_history>\nUSER: 你好\nASSISTANT: 你好！\n</chat_history>"
)


def test_openwebui_task_nonstream_bypasses_agent():
    """task 请求直接走 LLM，不调用 agent、不写 checkpoint。"""
    agent, llm = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": _OPENWEBUI_FOLLOWUPS_PROMPT}],
                },
            )
            assert r.status_code == 200
            body = r.json()
            assert body["choices"][0]["message"]["content"] == '{"follow_ups": ["q1?", "q2?"]}'
            # task 走 llm，不走 agent
            assert llm.ainvoke_calls == 1
            assert agent.ainvoke_calls == 0
    finally:
        app.dependency_overrides.clear()


def test_openwebui_task_stream_bypasses_agent():
    agent, llm = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": _OPENWEBUI_FOLLOWUPS_PROMPT}],
                    "stream": True,
                },
            )
            assert r.status_code == 200
            raw_lines = [ln.rstrip("\r") for ln in r.text.split("\n")]
            datas = [ln[6:] for ln in raw_lines if ln.startswith("data: ")]
            assert datas[-1] == "[DONE]"
            # 拼接所有 delta.content
            content_parts = []
            for d in datas[:-1]:
                c = json.loads(d)["choices"][0]["delta"].get("content")
                if c:
                    content_parts.append(c)
            assert "".join(content_parts) == '{"tags":[]}'
            # task 流式走 llm.astream，不走 agent
            assert llm.astream_calls == 1
            assert agent.astream_calls == 0
    finally:
        app.dependency_overrides.clear()


def test_openwebui_task_only_passes_last_message_to_llm():
    """task 请求即使带多条消息，也只把最后一条 task prompt 传给 LLM。"""
    agent, llm = _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [
                        {"role": "system", "content": "you are helpful"},
                        {"role": "user", "content": "你好"},
                        {"role": "assistant", "content": "你好！"},
                        {"role": "user", "content": _OPENWEBUI_FOLLOWUPS_PROMPT},
                    ],
                },
            )
            assert r.status_code == 200
            assert llm.ainvoke_calls == 1
            # LLM 只收到 1 条消息（最后一条 task prompt）
            assert len(llm.last_messages) == 1
            assert "### Task:" in llm.last_messages[0].content
    finally:
        app.dependency_overrides.clear()


# ==========================================
# 边界 + 回归
# ==========================================
def test_tool_role_rejected():
    """agent tools=[]，不支持 tool 角色，/v1/ 路径应返回 OpenAI 错误格式。"""
    _install_mocks()
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "tool", "content": "x"}],
                },
            )
            assert r.status_code == 400
            body = r.json()
            assert "error" in body
            assert "tool role messages are not supported" in body["error"]["message"]
            assert body["error"]["type"] == "api_error"
    finally:
        app.dependency_overrides.clear()


def test_models_listed():
    with TestClient(app) as client:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        ids = {m["id"] for m in body["data"]}
        assert "deepseek-chat" in ids


def test_items_route_still_works():
    """回归：原有 /items/{item_id} 路由不受影响。"""
    with TestClient(app) as client:
        r = client.get("/items/foo", headers={"X-Token": "coneofsilence"})
        assert r.status_code == 200
        assert r.json()["id"] == "foo"


# ==========================================
# thread_id 诊断日志（迭代 4 阶段 A）
# ==========================================
class _CaptureHandler(logging.Handler):
    """测试专用：把所有 LogRecord 收集到 list，便于断言。"""

    def __init__(self):
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _attach_capture(level: int) -> tuple[_CaptureHandler, logging.Logger]:
    """把 _CaptureHandler 挂到 thread_id logger，并打开 propagate 便于捕获。

    返回 (handler, logger)，调用方负责在 finally 里清理。
    """
    lg = logging.getLogger("thread_id")
    old_level = lg.level
    old_propagate = lg.propagate
    lg.setLevel(level)
    lg.propagate = True
    cap = _CaptureHandler()
    cap.setLevel(level)
    lg.addHandler(cap)

    def _restore():
        lg.removeHandler(cap)
        lg.setLevel(old_level)
        lg.propagate = old_propagate

    cap._restore = _restore  # type: ignore[attr-defined]
    return cap, lg


def test_thread_id_source_logged():
    """source 字段被记录到 thread_id logger，便于统计命中分布（INFO 级默认开）。"""
    agent, _ = _install_mocks()
    cap, _ = _attach_capture(logging.INFO)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "hi"}],
                    "metadata": {"chat_id": "win-diag"},
                },
            )
            assert r.status_code == 200

        resolved = [r.getMessage() for r in cap.records if "thread_id resolved" in r.getMessage()]
        assert len(resolved) == 1
        msg = resolved[0]
        assert "source=metadata.chat_id" in msg
        assert "value=win-diag" in msg
    finally:
        app.dependency_overrides.clear()
        cap._restore()


def test_thread_id_uuid_source_logged_when_nothing_provided():
    """messages 里无 user 消息（content-hash 也为空）时落到 uuid，source=uuid 被记录。"""
    agent, _ = _install_mocks()
    cap, _ = _attach_capture(logging.INFO)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "system", "content": "you are helpful"}]},
            )
            assert r.status_code == 200
        resolved = [r.getMessage() for r in cap.records if "thread_id resolved" in r.getMessage()]
        assert len(resolved) == 1
        assert "source=uuid" in resolved[0]
    finally:
        app.dependency_overrides.clear()
        cap._restore()


def test_thread_id_logger_does_not_leak_pii():
    """thread_id logger 的任何输出都不应含 Authorization/Cookie/消息 content 明文。

    迭代 4 阶段 A 的 DEBUG header 诊断块已移除（诊断完毕），现在 thread_id logger
    只产 INFO 级 `thread_id resolved` 日志，本测试守好 PII 防护底线。
    """
    agent, _ = _install_mocks()
    cap, _ = _attach_capture(logging.INFO)
    try:
        with TestClient(app) as client:
            r = client.post(
                "/v1/chat/completions",
                json={"model": "deepseek-chat",
                      "messages": [{"role": "user", "content": "secret-in-content"}]},
                headers={
                    "Authorization": "Bearer sk-super-secret-token-123",
                    "Cookie": "session=abc; csrf=xyz",
                },
            )
            assert r.status_code == 200

        blob = "\n".join(r.getMessage() for r in cap.records)
        # 任何敏感原文都不应入 thread_id logger
        assert "sk-super-secret-token-123" not in blob
        assert "session=abc" not in blob
        assert "secret-in-content" not in blob
    finally:
        app.dependency_overrides.clear()
        cap._restore()
