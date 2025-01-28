import os
import sys
import random
import operator
from typing import TypedDict, Annotated
from dataclasses import dataclass
from dotenv import load_dotenv
from pyprojroot import here

from datasets import Dataset
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Send
from langgraph.graph import END, START, StateGraph

from self_discover import prompts

load_dotenv()

langfuse_handler = CallbackHandler(
    public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
    secret_key=os.environ["LANGFUSE_SECRET_KEY"],
    host=os.environ["LANGFUSE_HOST"],
)


class SelfDiscoverState(TypedDict):
    task_descriptions: list[str]
    answer_formats: str
    task_examples: str
    selected_modules: str
    adapted_modules: str
    reasoning_structure: str
    reasonings: Annotated[list[str], operator.add]
    final_answers: list[str]


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

### Phase 1: Discovering Task-Specific Reasoning Structures ###


def select(state: SelfDiscoverState):
    chain = select_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {"selected_modules": result}


def adapt(state: SelfDiscoverState):
    chain = adapt_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {"adapted_modules": result}


def implement(state: SelfDiscoverState):
    chain = implement_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {"reasoning_structure": result}


### Phase 2: Reasoning with Task-Specific Reasoning Structures ###


def continue_to_reason(state: SelfDiscoverState):
    return [
        Send(
            "reason",
            {
                "task_description": task_description,
                "answer_formats": state["answer_formats"],
                "reasoning_structure": state["reasoning_structure"],
            },
        )
        for task_description in state["task_descriptions"]
    ]


def reason(state: PhaseIIState):
    chain = reasoning_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {"reasonings": [result]}


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

memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)


### Function to execute Self-Discover on a series of tasks
def self_discover(task_descriptions: list[str], model, answer_formats: str):
    n = len(task_descriptions)

    if not isinstance(model, BaseChatModel):
        raise ValueError(
            "model must be an instance of BaseChatModel. Are you sure you defined your LanggChain Chat Model properly."
        )

    if n == 0:
        raise ValueError("task_descriptions must be a non-empty list of strings.")

    Config.model = model
    state = {
        "task_descriptions": task_descriptions,
        "task_examples": "\n".join(random.sample(task_descriptions, min(3, n))),
        "answer_formats": answer_formats,
    }

    config = {"configurable": {"thread_id": 1}, "callbacks": [langfuse_handler]}
    for s in graph.stream(state, config=config):
        print(s)

    return graph.get_state(config).values
