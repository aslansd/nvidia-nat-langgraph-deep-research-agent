from datetime import datetime
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from deep_research.state import DeepAgentState
from deep_research.research_tools import tavily_search, think_tool
from deep_research.file_tools import ls, read_file, write_file
from deep_research.todo_tools import write_todos
from deep_research.task_tool import _create_task_tool
from deep_research.prompts import (
    FILE_USAGE_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    SUBAGENT_USAGE_INSTRUCTIONS,
    TODO_USAGE_INSTRUCTIONS,
)
from deep_research.research_tools import get_today_str

from functools import lru_cache

@lru_cache
def get_init_model():
    return init_chat_model()

def build_agent():
    """Build and return the LangGraph deep research agent."""

    model = get_init_model()

    max_concurrent_research_units = 3
    max_researcher_iterations = 3

    sub_agent_tools = [tavily_search, think_tool]
    built_in_tools = [ls, read_file, write_file, write_todos, think_tool]

    research_sub_agent = {
        "name": "research-agent",
        "description": "Delegate research to the sub-agent researcher.",
        "prompt": RESEARCHER_INSTRUCTIONS.format(date=get_today_str()),
        "tools": ["tavily_search", "think_tool"],
    }

    task_tool = _create_task_tool(
        sub_agent_tools,
        [research_sub_agent],
        model,
        DeepAgentState,
    )

    all_tools = sub_agent_tools + built_in_tools + [task_tool]

    SUBAGENT_INSTRUCTIONS = SUBAGENT_USAGE_INSTRUCTIONS.format(
        max_concurrent_research_units=max_concurrent_research_units,
        max_researcher_iterations=max_researcher_iterations,
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )

    INSTRUCTIONS = (
        "# TODO MANAGEMENT\n"
        + TODO_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# FILE SYSTEM USAGE\n"
        + FILE_USAGE_INSTRUCTIONS
        + "\n\n"
        + "=" * 80
        + "\n\n"
        + "# SUB-AGENT DELEGATION\n"
        + SUBAGENT_INSTRUCTIONS
    )

    agent = create_react_agent(
        model,
        all_tools,
        prompt=INSTRUCTIONS,
        state_schema=DeepAgentState,
    )

    return agent