# context.py
import contextvars
# 定义全局的 Request ID 上下文变量,在同一个请求中共享
request_id_ctx_var = contextvars.ContextVar("request_id", default="")

# RBAC 模块：当前请求的用户与租户上下文（由 auth_middleware / deps 写入）
tenant_id_ctx_var = contextvars.ContextVar("tenant_id", default="")
user_id_ctx_var = contextvars.ContextVar("user_id", default="")