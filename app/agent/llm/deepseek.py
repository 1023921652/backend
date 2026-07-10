import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_deepseek import ChatDeepSeek
load_dotenv(".env")
deepseek_llm = ChatDeepSeek(
    model=os.getenv("MODEL_NAME"),
    temperature=0,
    tags=['main_agent']
)