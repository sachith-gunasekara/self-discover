import os
import random
import operator
from typing import TypedDict, Annotated
from dataclasses import dataclass
from unittest import result
from dotenv import load_dotenv

from httpx import stream
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

load_dotenv()

from self_discover import prompts
from self_discover.helpers.logger import logger


langfuse_handler = CallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)
CONFIG = {"configurable": {"thread_id": 1}, "callbacks": [langfuse_handler]}


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


@dataclass
class Config:
    model: BaseChatModel


select_prompt = PromptTemplate.from_template(prompts.SELECT_PROMPT)
adapt_prompt = PromptTemplate.from_template(prompts.ADAPT_PROMPT)
implement_prompt = PromptTemplate.from_template(prompts.IMPLEMENT_PROMPT)
reasoning_prompt = PromptTemplate.from_template(prompts.REASONING_PROMPT)


def stream_response(chain, state):
    result = []

    for chunk in chain.stream(state):
        result.append(chunk)
        print(chunk, end="", flush=True)

    return "".join(result)


### Phase 1: Discovering Task-Specific Reasoning Structures ###


def select(state: SelfDiscoverState):
    logger.debug("Executing select function with state: {}", state.keys())
    chain = select_prompt | Config.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"selected_modules": result}


def adapt(state: SelfDiscoverState):
    logger.debug("Executing adapt function with state: {}", state.keys())
    chain = adapt_prompt | Config.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"adapted_modules": result}


def implement(state: SelfDiscoverState):
    logger.debug("Executing implement function with state: {}", state.keys())
    chain = implement_prompt | Config.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning_structure": result}


### Phase 2: Reasoning with Task-Specific Reasoning Structures ###


def dummy_node(state: SelfDiscoverState):
    logger.debug("Executing dummy_node function with state: {}", state.keys())
    logger.debug("Dummy node state: {}", state.keys())
    pass


def continue_to_reason(state: SelfDiscoverState):
    logger.debug("Executing continue_to_reason function with state: {}", state.keys())
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
    logger.debug("Executing reason function with state: {}", state.keys())
    chain = reasoning_prompt | Config.model | StrOutputParser()

    result = stream_response(chain, state)

    return {"reasoning": result}


### Compile graph with memory saver ###


def compile_graph(graph: StateGraph):
    logger.debug("Compiling graph: {}", graph)
    memory = MemorySaver()

    compiled_graph = graph.compile(checkpointer=memory)
    logger.debug("Compiled graph: {}", compiled_graph)
    return compiled_graph


### Self-Discover Graph ###


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


### Self-Discover Phase I Graph ###


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


### Self-Discover Phase II Graph ###


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


def validate_params(
    task_description: list[str], model: BaseChatModel, answer_formats: str
):
    logger.debug("Validating parameters")

    if not isinstance(model, BaseChatModel):
        logger.error("Invalid model type: {}", type(model))
        raise ValueError(
            "The model must be an instance of BaseChatModel. Ensure you have defined your LangChain Chat Model properly."
        )

    if not task_description or not all(
        isinstance(task, str) for task in task_description
    ):
        logger.error("Invalid task_description: {}", task_description)
        raise ValueError("task_description must be a non-empty list of strings.")

    if not isinstance(answer_formats, str):
        logger.error("Invalid answer_formats type: {}", type(answer_formats))
        raise ValueError("answer_formats must be a string.")


def invoke_graph(graph, state, config, stream):
    logger.info(
        "Invoking graph with state: {}, config: {}, stream: {}",
        state.keys(),
        config,
        stream,
    )
    if stream:
        for s in graph.stream(state, config=config):
            logger.debug(s)

        return graph.get_state(config).values
    else:
        result = graph.invoke(state, config=config)
        return result


def call_graph(state, phase, stream):
    logger.info(
        "Calling graph with state: {}, phase: {}, stream: {}",
        state.keys(),
        phase,
        stream,
    )

    if phase == -1:
        graph = build_self_discover_graph()
    elif phase == 1:
        graph = build_self_discover_phaseI_graph()
    elif phase == 2:
        graph = build_self_discover_phaseII_graph()
    else:
        logger.error("Invalid phase: {}", phase)
        raise ValueError("Invalid phase. Must be 0, 1, or 2.")

    return invoke_graph(graph, state, CONFIG, stream)


### Function to execute Self-Discover on a series of tasks
def self_discover(
    task_description: list[str],
    model: BaseChatModel,
    reasoning_structure: str,
    answer_formats: str = "",
    phase: int = -1,
    stream: bool = False,
):
    logger.info("Starting self_discover")
    validate_params(task_description, model, answer_formats)

    Config.model = model

    if phase in (-1, 1):
        state = {
            "task_description": task_description,
            "task_examples": "\n\n".join(
                random.sample(task_description, min(10, len(task_description)))
            ),
            "answer_formats": answer_formats,
        }
    elif phase == 2:
        state = {
            "task_description": task_description,
            "answer_formats": answer_formats,
            "reasoning_structure": reasoning_structure,
        }

    result = call_graph(state, phase, stream)
    logger.info("self_discover result: {}", result.keys())
    return result
