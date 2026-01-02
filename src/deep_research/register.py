from pydantic import BaseModel, Field
from nat.cli.register_workflow import register_function
from nat.builder.function_info import FunctionInfo
from nat.builder.builder import Builder
from nat.data_models.function import FunctionBaseConfig

from deep_research.graph import build_agent

# ---- Input schema ----
class DeepResearchInput(BaseModel):
    question: str = Field(description="Research question to investigate deeply")

# ---- Config (THIS NAME MATTERS) ----
class DeepResearchConfig(
    FunctionBaseConfig,
    name="run_deep_research_agent",
):
    pass

# ---- Registration ----
@register_function(config_type=DeepResearchConfig)
async def deep_research_tool(config: DeepResearchConfig, builder: Builder):
    # ⚠️ DO NOT build agent here
    agent = None

    async def _wrapper(question: str) -> str:
        nonlocal agent
        if agent is None:
            agent = build_agent()  # lazy init

        result = agent.invoke({
            "messages": [("user", question)]
        })
        return result["messages"][-1].content

    yield FunctionInfo.from_fn(
        _wrapper,
        input_schema=DeepResearchInput,
        description="Run a multi-step deep research agent using LangGraph",
    )