from pydantic import BaseModel
from typing import AsyncGenerator, Optional, List, Dict, Any, NotRequired
from langchain.agents import AgentState

class CustomAgentState(AgentState):
    pass