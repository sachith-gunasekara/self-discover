# Directly adapted and modified from https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover

import re
from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

from ._prompts import _iself_discover_prompts as prompts
from ._helpers.logger import logger
from ._helpers.validators import validate_params
from ._helpers import stream_response, invoke_graph
from ._config import LLM, Langfuse


## Initial Prompts ##

select_prompt = PromptTemplate.from_template(prompts.SELECT_PROMPT)
adapt_prompt = PromptTemplate.from_template(prompts.ADAPT_PROMPT)

## Reason Prompts ##

# Structured #
structured_planning_prompt = PromptTemplate.from_template(
    prompts.STRUCTURED_PLANNING_PROMPT
)
structured_application_prompt = PromptTemplate.from_template(
    prompts.STRUCTURED_APPLICATION_PROMPT
)

# Unstructured #
unstructured_planning_prompt = PromptTemplate.from_template(
    prompts.UNSTRUCTURED_PLANNING_PROMPT
)
unstructured_application_prompt = PromptTemplate.from_template(
    prompts.UNSTRUCTURED_APPLICATION_PROMPT
)


class ISelfDiscoverState(TypedDict):
    task_description: str
    answer_formats: str
    few_shot_examples: str
    task_description_backup: Optional[str]
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_plan: Optional[str]
    reasoning: Optional[str]


## Corrector nodes for few shot examples ##


def append_few_shot_examples(state: ISelfDiscoverState):
    logger.info("Appending few shot examples to task description")

    task_description = state["task_description"]

    return {
        "task_description": state["task_description"]
        + f"\n\n{prompts.FEW_SHOT_EXAMPLES_PROMPT.format(few_shot_examples_str=state['few_shot_examples'])}",
        "task_description_backup": task_description,
    }


def restore_task_description(state: ISelfDiscoverState):
    logger.info("Restoring original task description before APPLICATION step")

    return {"task_description": state["task_description_backup"]}


## Initial Nodes ##


def select(state: ISelfDiscoverState):
    logger.info("Executing SELECT step")
    chain = select_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"selected_modules": result}


def adapt(state: ISelfDiscoverState):
    logger.info("Executing ADAPT step")
    chain = adapt_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"adapted_modules": result}  # -> reasoning


## Reason Nodes ##

# Structured #


def structured_planning(state: ISelfDiscoverState):
    logger.info("Executing STRUCTURED PLANNING step")
    chain = structured_planning_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_plan": result}


def structured_application(state: ISelfDiscoverState):
    logger.info("Executing STRUCTURED APPLICATION step")
    chain = structured_application_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": result}


# Unstructured #


def unstructured_planning(state: ISelfDiscoverState):
    logger.info("Executing UNSTRUCTURED PLANNING step")
    chain = unstructured_planning_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_plan": result}


def unstructured_application(state: ISelfDiscoverState):
    logger.info("Executing UNSTRUCTURED APPLICATION step")
    chain = unstructured_application_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": result}


## Routing Functions ##


def determine_initial_node(state: ISelfDiscoverState):
    if state["few_shot_examples"]:
        return "append_few_shot_examples"
    else:
        return "select"


def determine_pre_application_node(state: ISelfDiscoverState):
    if state["few_shot_examples"]:
        return "restore_task_description"
    else:
        return "application"


def build_iself_discover_graph(structured: bool = False):
    logger.info("Building iSelf-Discover graph")
    graph_builder = StateGraph(ISelfDiscoverState)

    graph_builder.add_node(append_few_shot_examples)
    graph_builder.add_node(restore_task_description)
    graph_builder.add_node(select)
    graph_builder.add_node(adapt)

    if structured:
        graph_builder.add_node("planning", structured_planning)
        graph_builder.add_node("application", structured_application)
    else:
        graph_builder.add_node("planning", unstructured_planning)
        graph_builder.add_node("application", unstructured_application)

    graph_builder.add_conditional_edges(START, determine_initial_node)
    graph_builder.add_edge("append_few_shot_examples", "select")
    graph_builder.add_edge("select", "adapt")
    graph_builder.add_edge("adapt", "planning")
    graph_builder.add_conditional_edges("planning", determine_pre_application_node)
    graph_builder.add_edge("restore_task_description", "application")
    graph_builder.add_edge("application", END)

    logger.debug("Compiling iSelf-Discover graph")
    return graph_builder.compile()


def iself_discover(
    task_description: str,
    model: BaseChatModel,
    answer_formats: str,
    structured: bool = False,
    few_shot_examples_str: str = "",
    stream: bool = False,
):
    logger.info("Starting iSelf-Discover")
    validate_params(task_description, model, answer_formats)

    LLM.model = model

    state = {
        "task_description": task_description,
        "answer_formats": answer_formats,
        "few_shot_examples": few_shot_examples_str,
    }

    graph = build_iself_discover_graph(structured)

    result = invoke_graph(graph, state, Langfuse.CONFIG, stream)
    logger.debug("iSelf-Discover result: {}", result.keys())
    return result
