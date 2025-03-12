# Directly adapted and modified from https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover

import re
from typing import Optional, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()

from ._prompts import _phased_self_discover_prompts as prompts
from ._helpers.handlers import langfuse_handler
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


class PhasedSelfDiscoverState(TypedDict):
    task_description: str
    answer_formats: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_plan: Optional[str]
    reasoning: Optional[str]


## Initial Nodes ##


def select(state: PhasedSelfDiscoverState):
    logger.info("Executing SELECT step")
    chain = select_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"selected_modules": result}


def adapt(state: PhasedSelfDiscoverState):
    logger.info("Executing ADAPT step")
    chain = adapt_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"adapted_modules": result}  # -> reasoning


## Reason Nodes ##

# Structured #


def structured_planning(state: PhasedSelfDiscoverState):
    logger.info("Executing STRUCTURED PLANNING step")
    chain = structured_planning_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_plan": result}


def structured_application(state: PhasedSelfDiscoverState):
    logger.info("Executing STRUCTURED APPLICATION step")
    chain = structured_application_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": result}


# Unstructured #


def unstructured_planning(state: PhasedSelfDiscoverState):
    logger.info("Executing UNSTRUCTURED PLANNING step")
    chain = unstructured_planning_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_plan": result}


def unstructured_application(state: PhasedSelfDiscoverState):
    logger.info("Executing UNSTRUCTURED APPLICATION step")
    chain = unstructured_application_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": result}


def build_phased_self_discover_graph(structured: bool = False):
    logger.info("Building Phased Self-Discover graph")
    graph_builder = StateGraph(PhasedSelfDiscoverState)

    graph_builder.add_node(select)
    graph_builder.add_node(adapt)

    graph_builder.add_edge(START, "select")
    graph_builder.add_edge("select", "adapt")

    if structured:
        graph_builder.add_node(structured_planning)
        graph_builder.add_node(structured_application)

        graph_builder.add_edge("adapt", "structured_planning")
        graph_builder.add_edge("structured_planning", "structured_application")
        graph_builder.add_edge("structured_application", END)
    else:
        graph_builder.add_node(unstructured_planning)
        graph_builder.add_node(unstructured_application)

        graph_builder.add_edge("adapt", "unstructured_planning")
        graph_builder.add_edge("unstructured_planning", "unstructured_application")
        graph_builder.add_edge("unstructured_application", END)

    logger.debug("Compiling Phased Self-Discover graph")
    return graph_builder.compile()


def phased_self_discover(
    task_description: str,
    model: BaseChatModel,
    answer_formats: str,
    structured: bool = False,
    stream: bool = False
):
    logger.info("Starting Phased Self-Discover")
    validate_params(task_description, model, answer_formats)

    LLM.model = model

    state = {"task_description": task_description, "answer_formats": answer_formats}

    graph = build_phased_self_discover_graph(structured)

    result = invoke_graph(graph, state, Langfuse.CONFIG, stream)
    logger.debug("Phased Self-Discover result: {}", result.keys())
    return result
