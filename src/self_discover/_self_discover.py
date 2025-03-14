import random
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph


load_dotenv()


from ._prompts import _self_discover_prompts as prompts
from ._helpers.logger import logger
from ._helpers.enums import Phase
from ._helpers.validators import validate_params
from ._helpers import stream_response, invoke_graph
from ._config import LLM, Langfuse


class SelfDiscoverState(TypedDict):
    task_description: list[str]
    answer_formats: str
    task_examples: str
    selected_modules: str
    adapted_modules: str
    reasoning_structure: str
    reasoning: Annotated[list[str], operator.add]


class PhaseIIState(TypedDict):
    task_description: str
    answer_formats: str
    reasoning_structure: str


select_prompt = PromptTemplate.from_template(prompts.SELECT_PROMPT)
adapt_prompt = PromptTemplate.from_template(prompts.ADAPT_PROMPT)
implement_prompt = PromptTemplate.from_template(prompts.IMPLEMENT_PROMPT)
reasoning_prompt = PromptTemplate.from_template(prompts.REASONING_PROMPT)


## Phase 1: Discovering Task-Specific Reasoning Structures ##


def select(state: SelfDiscoverState):
    logger.info("Executing SELECT step")
    chain = select_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"selected_modules": result}


def adapt(state: SelfDiscoverState):
    logger.info("Executing ADAPT step")
    chain = adapt_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"adapted_modules": result}


def implement(state: SelfDiscoverState):
    logger.info("Executing IMPLEMENT step")
    chain = implement_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_structure": result}


## Phase 2: Reasoning with Task-Specific Reasoning Structures ##


def dummy_node(state: SelfDiscoverState):
    pass


def continue_to_reason(state: SelfDiscoverState):
    logger.info("Sending tasks to REASON step from continue_to_reason")
    return [
        Send(
            "reason",
            {
                "task_description": task_description,
                "answer_formats": state["answer_formats"],
                "reasoning_structure": state["reasoning_structure"],
            },
        )
        for task_description in state["task_description"]
    ]


def reason(state: PhaseIIState):
    logger.debug("Executing REASON step")
    chain = reasoning_prompt | LLM.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": [result]}


## Compile graph with memory saver ##


def compile_graph(graph: StateGraph):
    logger.debug("Compiling Self-Discover graph")
    memory = MemorySaver()

    compiled_graph = graph.compile(checkpointer=memory)
    return compiled_graph


## Self-Discover Graph ##


def build_self_discover_graph():
    logger.info("Building self-discover graph")
    graph_builder = StateGraph(SelfDiscoverState)

    ## Phase I
    graph_builder.add_node(select)
    graph_builder.add_node(adapt)
    graph_builder.add_node(implement)

    graph_builder.add_edge(START, "select")
    graph_builder.add_edge("select", "adapt")
    graph_builder.add_edge("adapt", "implement")

    ## Phase II
    graph_builder.add_node(reason)

    graph_builder.add_conditional_edges("implement", continue_to_reason, ["reason"])
    graph_builder.add_edge("reason", END)

    return compile_graph(graph_builder)


## Self-Discover Phase I Graph ##


def build_self_discover_phaseI_graph():
    logger.info("Building self-discover Phase I graph")
    graph_builder = StateGraph(SelfDiscoverState)

    ## Phase I
    graph_builder.add_node(select)
    graph_builder.add_node(adapt)
    graph_builder.add_node(implement)

    graph_builder.add_edge(START, "select")
    graph_builder.add_edge("select", "adapt")
    graph_builder.add_edge("adapt", "implement")

    return compile_graph(graph_builder)


## Self-Discover Phase II Graph ##


def build_self_discover_phaseII_graph():
    logger.info("Building self-discover Phase II graph")
    graph_builder = StateGraph(SelfDiscoverState)

    ## Phase II
    graph_builder.add_node(dummy_node)
    graph_builder.add_node(reason)

    graph_builder.add_edge(START, "dummy_node")
    graph_builder.add_conditional_edges("dummy_node", continue_to_reason, ["reason"])
    graph_builder.add_edge("reason", END)

    return compile_graph(graph_builder)


def call_graph(state, phase, stream):
    logger.info(
        "Calling graph with phase: {}, stream: {}",
        phase,
        stream,
    )

    if phase == Phase.BOTH.value:
        graph = build_self_discover_graph()
    elif phase == Phase.I.value:
        graph = build_self_discover_phaseI_graph()
    elif phase == Phase.II.value:
        graph = build_self_discover_phaseII_graph()
    else:
        logger.error("Invalid phase: {}", phase)
        raise ValueError(
            f"Invalid phase. Must be {Phase.BOTH.value}, {Phase.I.value}, or {Phase.II.value}."
        )

    return invoke_graph(graph, state, Langfuse.CONFIG, stream)


## Function to execute Self-Discover on a series of tasks ##


def self_discover(
    task_description: list[str],
    model: BaseChatModel,
    answer_formats: str,
    reasoning_structure: str = "",
    phase: int = Phase.BOTH.value,
    stream: bool = False,
):
    logger.info("Starting Self-Discover")
    validate_params(task_description, model, answer_formats)

    LLM.model = model

    if phase in (Phase.BOTH.value, Phase.I.value):
        state = {
            "task_description": task_description,
            "task_examples": "\n\n".join(
                random.sample(task_description, min(10, len(task_description)))
            ),
            "answer_formats": answer_formats,
        }
    elif phase == Phase.II.value:
        state = {
            "task_description": task_description,
            "answer_formats": answer_formats,
            "reasoning_structure": reasoning_structure,
        }

    result = call_graph(state, phase, stream)
    logger.debug("Self-Discover result: {}", result.keys())
    return result
