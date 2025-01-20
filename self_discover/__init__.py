from typing import TypedDict
from dataclasses import dataclass
from dotenv import load_dotenv
from pyprojroot import here

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langfuse.callback import CallbackHandler
from langgraph.graph import END, START, StateGraph

from self_discover import prompts


class SelfDiscoverState(TypedDict):
    task_descriptions: list[str]
    reasoning_structure: str
    answer_formats: str

class PhaseIState(TypedDict):
    task_examples: str
    selected_modules: str
    adapted_modules: str
    reasoning_structure: str

class PhaseIIState(TypedDict):
    task_description: str
    answer_formats: str
    reasoning_structure: str
    reasoning: str

@dataclass
class Config:
    model: BaseChatModel


select_prompt = PromptTemplate.from_template(prompts.SELECT_PROMPT)
adapt_prompt = PromptTemplate.from_template(prompts.ADAPT_PROMPT)
implement_prompt = PromptTemplate.from_template(prompts.IMPLEMENT_PROMPT)
reasoning_prompt = PromptTemplate.from_template(prompts.REASONING_PROMPT)

### Phase 1: Discovering Task-Specific Reasoning Structures ###

def select(state: PhaseIState):
    chain = select_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {
        "selected_modules": result
    }

def adapt(state: PhaseIState):
    chain = adapt_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {
        "adapted_modules": result
    }

def implement(state: PhaseIState):
    chain = implement_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {
        "reasoning_structure": result
    }

phaseI_graph_builder = StateGraph(PhaseIState)

phaseI_graph_builder.add_node(select)
phaseI_graph_builder.add_node(adapt)
phaseI_graph_builder.add_node(implement)

phaseI_graph_builder.add_edge(START, "select")
phaseI_graph_builder.add_edge("select", "adapt")
phaseI_graph_builder.add_edge("adapt", "implement")

phaseI_graph = phaseI_graph_builder.compile()

### Phase 2: Reasoning with Task-Specific Reasoning Structures ###

def reason(state: PhaseIIState):
    chain = reasoning_prompt | Config.model | StrOutputParser()

    result = chain.invoke(state)

    return {
        "reasoning_structure": result
    }

phaseII_graph_builder = StateGraph(PhaseIIState)

phaseII_graph_builder.add_node(reason)

phaseII_graph_builder.add_edge(START, "reason")

phaseII_graph = phaseII_graph_builder.compile()

### Self-Discover ###

self_discover_graph_builder = StateGraph(SelfDiscoverState)

self_discover_graph_builder.add_node("pI", phaseI_graph)
self_discover_graph_builder.add_node("pII", phaseII_graph)

self_discover_graph_builder.add_edge(START, "pI")
self_discover_graph_builder.add_edge("pI", "pII")
self_discover_graph_builder.add_edge("pII", END)

self_discover_graph = self_discover_graph_builder.compile()


### Function to execute Self-Discover on a series of tasks
def self_discover(task_descriptions: list[str], model, answer_formats: str):
    if not isinstance(model, BaseChatModel):
        raise ValueError("model must be an instance of BaseChatModel. Are you sure you defined your LanggChain Chat Model properly.")
    
    state = {
        "task_descriptions": task_descriptions,
        "answer_formats": answer_formats
    }

    return self_discover_graph.invoke(state)