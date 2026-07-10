# context.py
import contextvars
# 定义全局的 Request ID 上下文变量,在同一个请求中共享
request_id_ctx_var = contextvars.ContextVar("request_id", default="")