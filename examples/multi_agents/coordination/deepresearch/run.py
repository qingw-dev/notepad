
import os

from aworld.agents.llm_agent import Agent
from aworld.config.conf import AgentConfig, ModelConfig
from aworld.core.agent.swarm import TeamSwarm
from aworld.core.event.base import Constants
from aworld.planner.plan import PlannerOutputParser
from aworld.runner import Runners

from examples.common.tools.common import Tools
from examples.multi_agents.coordination.deepresearch.prompts import *

# os.environ["LLM_MODEL_NAME"] = "DeepSeek-V3"
# os.environ["LLM_MODEL_NAME"] = "qwen/qwen3-8b"
# os.environ["LLM_BASE_URL"] = "YOUR_LLM_BASE_URL"
# os.environ["LLM_API_KEY"] = "YOUR_LLM_API_KEY"

def get_deepresearch_swarm(user_input):

    agent_config = AgentConfig(
        llm_config=ModelConfig(
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            llm_model_name=os.getenv("LLM_MODEL_NAME"),
            llm_base_url=os.getenv("LLM_BASE_URL"),
            llm_api_key=os.getenv("LLM_API_KEY"),
            llm_temperature=os.getenv("LLM_TEMPERATURE", 0.0)
        ),
        use_vision=False
    )

    agent_id = "planner_agent"
    plan_agent = Agent(
        agent_id = agent_id,
        name="planner_agent",
        desc="planner_agent",
        conf=agent_config,
        use_tools_in_prompt=True,
        resp_parse_func=PlannerOutputParser(agent_id).parse,
        system_prompt_template=plan_sys_prompt,
        response_handler_name=Constants.PLAN
    )

    web_search_agent = Agent(
        name="web_search_agent",
        desc="web_search_agent",
        conf=agent_config,
        system_prompt_template=search_sys_prompt,
        tool_names=[Tools.SEARCH_API.value]
    )
    
    reporting_agent = Agent(
        name="reporting_agent",
        desc="reporting_agent",
        conf=agent_config,
        system_prompt_template=reporting_sys_prompt,
    )

    return TeamSwarm(plan_agent, web_search_agent, reporting_agent, max_steps=1)
    

if __name__ == "__main__":
    user_input = "7天北京旅游计划"
    swarm = get_deepresearch_swarm(user_input)
    result = Runners.sync_run(
        input=user_input,
        swarm=swarm
    )
    print("deepresearch result: ", result)
