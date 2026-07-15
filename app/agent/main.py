from app.agent.llm.deepseek import deepseek_llm
from langchain.agents import create_agent
from app.agent.config.redis_config import get_redis_checkpointer
from app.agent.state.main_state import CustomAgentState
from app.agent.llm import get_llm
from app.agent.middleware.context import OrphanToolCallSanitizerMiddleware
# 定义一个异步工厂函数 / 依赖项
async def set_agent(mcp_tools: list = None):
    memory = await get_redis_checkpointer()
    tools = list(mcp_tools or [])
    agent = create_agent(
        model=get_llm(),
        tools=tools,
        state_schema=CustomAgentState,
        middleware=[OrphanToolCallSanitizerMiddleware()],
        checkpointer=memory,
        name="main_agent",
        system_prompt="""
                        你是一个高效的 AI 助手。请严格遵守以下准则：
                            1. [核心任务]：仅根据用户提出的最新问题进行回答。
                            2. [历史处理]：仅在对话确实需要引用历史背景（如：上下文指代、补充说明）时，才提及历史信息。除非用户询问，严禁主动输出总结、流程分析、意图提取或无关的文学创作。
                            3. [输出格式]：直接回答问题，不要添加任何结构化标签（如SESSION INTENT, SUMMARY等）。
                            4. [身份限制]：当询问你是谁、什么模型时，仅回复“我是你的AI助手”。
                            5. [简洁原则]：内容精炼，拒绝无关解释。
                            当用户输入/rag开头时，必须调用合适的rag工具，一个工具未返回结果时或数据不相关时调用另外的工具，
                            如果没有相关的数据，返回未检索到相关数据，严禁编造错误数据
                      """
    )
    return agent




