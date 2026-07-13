import os
from langchain_deepseek import ChatDeepSeek

# .env 已由 app.main 在启动早期 load_dotenv；这里不重复 load，
# 避免每个模块各自 load 造成的副作用（cwd 依赖、重复 IO）。
deepseek_llm = ChatDeepSeek(
    model=os.getenv("MODEL_NAME", "deepseek-chat"),
    temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    tags=['main_agent']
)