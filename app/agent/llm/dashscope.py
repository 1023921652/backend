import os
from langchain_openai import ChatOpenAI

# .env 已由 app.main 在启动早期 load_dotenv；这里不重复 load。
# 命名规范与 deepseek.py 一致：模块导出 {provider}_llm，
# —— registry 用 getattr(module, "dashscope_llm") 取实例
dashscope_llm = ChatOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv(
        "DASHSCOPE_BASE_URL",
        "https://ws-oi8z1umy0fuyv6if.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    ),
    model=os.getenv("DASHSCOPE_MODEL_NAME", "qwen3.6-plus"),
    temperature=float(os.getenv("DASHSCOPE_TEMPERATURE", "0")),
    tags=["main_agent"],
)
