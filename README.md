Swagger 一键测试流程：
  1. 打开 http://localhost:8000/docs
  2. 找 POST /v1/auth/bootstrap → Try it out → 填 body：
  {
    "username": "alice",
    "email": "alice@x.com",
    "password": "Alice1234",
    "enterprise_name": "Acme Inc",
    "enterprise_slug": "acme"
  }
  2. → Execute → 复制响应里的 access_token
  3. 右上角 Authorize 🔒 → 粘贴 access_token（不带 Bearer  前缀）→ Authorize
  4. 测试 /me、/enterprises/{eid}/members、/enterprises/{eid}/roles 等 13 个带锁端点，自动带 Authorization 头

  Bootstrap 内部在一个事务里串了三步：register → create_enterprise（自动产出 owner/admin/member 角色 + 把创建者设为 owner）→ login_enterprise（签发
  token）。生产环境建议按 ENV 关掉。


步骤2返回


{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwidGVuYW50X2lkIjoxLCJpYXQiOjE3ODQyNzgyNjEsImV4cCI6MTc4NDI4MDA2MSwianRpIjoiN2I0NDBiM2UtOWQzNi00ODVjLWJlODAtNGM5Nzk1NjQ0ZjNjIiwidHlwZSI6ImFjY2VzcyIsInJvbGVzIjpbIm93bmVyIl0sInNjb3BlcyI6WyJhdWRpdDpyZWFkIiwiZW50ZXJwcmlzZTpjcmVhdGUiLCJlbnRlcnByaXNlOmRlbGV0ZSIsImVudGVycHJpc2U6cmVhZCIsImVudGVycHJpc2U6dXBkYXRlIiwibWVtYmVyOmludml0ZSIsIm1lbWJlcjpyZWFkIiwibWVtYmVyOnJlbW92ZSIsIm1lbWJlcjp1cGRhdGUiLCJwZXJtaXNzaW9uOnJlYWQiLCJyb2xlOmNyZWF0ZSIsInJvbGU6ZGVsZXRlIiwicm9sZTpyZWFkIiwicm9sZTp1cGRhdGUiLCJ1c2VyOmNyZWF0ZSIsInVzZXI6ZGVsZXRlIiwidXNlcjpyZWFkIiwidXNlcjp1cGRhdGUiXX0.lH5WlwucpHnCl4b7N2kt9bD_wAoNc4Zi02GgbO0uv_0",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwidGVuYW50X2lkIjoxLCJpYXQiOjE3ODQyNzgyNjEsImV4cCI6MTc4NDg4MzA2MSwianRpIjoiZTg1MTNjMWEtY2EzMi00NjFiLWE3YzAtNGUwYWNmZjY2NTFjIiwidHlwZSI6InJlZnJlc2gifQ.n51x56_CjGgI9veKyjoQVoqAfhHnFiN_-aB9ym3Yel8",
  "token_type": "Bearer",
  "expires_in": 1800,
  "scopes": [
    "audit:read",
    "enterprise:create",
    "enterprise:delete",
    "enterprise:read",
    "enterprise:update",
    "member:invite",
    "member:read",
    "member:remove",
    "member:update",
    "permission:read",
    "role:create",
    "role:delete",
    "role:read",
    "role:update",
    "user:create",
    "user:delete",
    "user:read",
    "user:update"
  ],
  "roles": [
    "owner"
  ]
}
