# Directly adapted and modified from https://langchain-ai.github.io/langgraph/tutorials/self-discover/self-discover

import re
from typing import Optional, TypedDict

from langchain import hub
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv
import pandas as pd
from pyprojroot import here
from langfuse.callback import CallbackHandler

load_dotenv()

from struct_vs_unstruct.helpers.llm import model
from struct_vs_unstruct.helpers.log import log_token_usage
import struct_vs_unstruct.prompts as svu_prompts


langfuse_handler = CallbackHandler(    
    public_key="",
    secret_key="",
    host="https://cloud.langfuse.com"
)


select_prompt = PromptTemplate.from_template(svu_prompts.SELECT_PROMPT)
adapt_prompt = PromptTemplate.from_template(svu_prompts.ADAPT_PROMPT)
structured_prompt = PromptTemplate.from_template(svu_prompts.STRUCTURING_PROMPT)
reasoning_prompt = PromptTemplate.from_template(svu_prompts.REASONING_PROMPT)

deriving_reasoning_modules_prompt = PromptTemplate.from_template(svu_prompts.DERIVING_REASONING_MODULES_PROMPT)
nl_reasoning_plan_prompt = PromptTemplate.from_template(svu_prompts.NL_REASONING_PLAN_PROMPT)
follow_reasoning_plan_prompt = PromptTemplate.from_template(svu_prompts.FOLLOW_REASONING_PLAN_PROMPT)
structure_response_prompt = PromptTemplate.from_template(svu_prompts.STRUCTURE_RESPONSE_PROMPT)


class SelfDiscoverState(TypedDict):
    reasoning_modules: str
    task_description: str
    reasoning_formats: str
    selected_modules: Optional[str]
    adapted_modules: Optional[str]
    reasoning_structure: Optional[str]
    reasoning_plan: Optional[str]
    reasoning: Optional[str]
    trajectory: Optional[str]
    answer_pred: Optional[str]

def output_parser(result, json: bool = False):
    if json:
        parser = JsonOutputParser()
    else:
        parser = StrOutputParser()
    return parser.invoke(result)

def select(inputs):
    select_chain = select_prompt | model
    
    result = select_chain.invoke(inputs)

    log_token_usage(result)

    return {"selected_modules": output_parser(result)}


def adapt(inputs):
    adapt_chain = adapt_prompt | model

    result = adapt_chain.invoke(inputs)

    log_token_usage(result)

    return {"adapted_modules": output_parser(result)} # -> reasoning 


def structure(inputs):
    structure_chain = structured_prompt | model

    result = structure_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning_structure": output_parser(result)}


def reason(inputs):
    reasoning_chain = reasoning_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning": output_parser(result)}

def deriving_reasoning_modules(inputs):
    reasoning_chain = deriving_reasoning_modules_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"selected_modules": output_parser(result), "adapted_modules": output_parser(result)}

def nl_reasoning_plan(inputs):
    reasoning_chain = nl_reasoning_plan_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning_plan": output_parser(result)}


def follow_reasoning_plan(inputs):
    reasoning_chain = follow_reasoning_plan_prompt | model

    result = reasoning_chain.invoke(inputs)

    log_token_usage(result)

    return {"reasoning": output_parser(result)}

def structure_response_with_llm(inputs):
    structure_chain = structure_response_prompt | model

    result = structure_chain.invoke(inputs)

    log_token_usage(result)

    response_json = output_parser(result, json=True)

    if "answer_pred" not in response_json:
        response_json["answer_pred"] = None

    return response_json

def structure_response_without_llm(inputs):
    text = "The final answer is "
    pattern = fr"(?<={text}).*"

    response = inputs["reasoning"]

    try:
        answer, trajectory = re.search(pattern, response).group(0).strip(), re.sub(pattern, "", response).replace(text, "").strip()
    except:
        answer, trajectory = None, response

    return {
        "trajectory": trajectory,
        "answer_pred": answer
    }


def add_nodes(modified: bool = False, structure_with_llm: bool = False, self_synthesis: bool = False):
    graph = StateGraph(SelfDiscoverState)

    if not modified:
        graph.add_node(select)
        graph.add_node(adapt)
        graph.add_node(structure)
        graph.add_node(reason)
    else:
        if not self_synthesis:
            graph.add_node(select)
            graph.add_node(adapt)
        else:
            graph.add_node(deriving_reasoning_modules)
        graph.add_node(nl_reasoning_plan)
        graph.add_node(follow_reasoning_plan)

    if structure_with_llm:
        graph.add_node(structure_response_with_llm)
    else:
        graph.add_node(structure_response_without_llm)

    return graph


def create_self_discover_graph(modified: bool = False, structure_with_llm: bool = False, self_synthesis: bool = False):
    graph = add_nodes(modified, structure_with_llm, self_synthesis)

    if not modified:
        graph.add_edge(START, "select")
        graph.add_edge("select", "adapt")
        graph.add_edge("adapt", "structure")
        graph.add_edge("structure", "reason")
        
        if structure_with_llm:
            graph.add_edge("reason", "structure_response_with_llm")
            graph.add_edge("structure_response_with_llm", END)
        else:
            graph.add_edge("reason", "structure_response_without_llm")
            graph.add_edge("structure_response_without_llm", END)
    else:
        if not self_synthesis:
            graph.add_edge(START, "select")
            graph.add_edge("select", "adapt")
            graph.add_edge("adapt", "nl_reasoning_plan")
        else:
            graph.add_edge(START, "deriving_reasoning_modules")
            graph.add_edge("deriving_reasoning_modules", "nl_reasoning_plan")

        graph.add_edge("nl_reasoning_plan", "follow_reasoning_plan")

        if structure_with_llm:
            graph.add_edge("follow_reasoning_plan", "structure_response_with_llm")
            graph.add_edge("structure_response_with_llm", END)
        else:
            graph.add_edge("follow_reasoning_plan", "structure_response_without_llm")
            graph.add_edge("structure_response_without_llm", END)

    app = graph.compile()

    return app


def phased_self_discover(task_description: str, reasoning_formats: str, modified: bool = False, structure_with_llm: bool = False, self_synthesis: bool = False):
    reasoning_modules = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        # "3. How could I measure progress on this problem?",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "6. What are the potential risks and drawbacks of each solution?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        # "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        # "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
        "35. Let’s imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
        "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
        "37. Ignoring the current best solution, create an entirely new solution to the problem."
        # "38. Let’s think step by step."
        "39. Let’s make a step by step plan and implement it with good notation and explanation.",
    ]
    reasoning_modules_str = "\n".join(reasoning_modules)

    app = create_self_discover_graph(modified, structure_with_llm, self_synthesis)

    return app.invoke(
        {
            "task_description": task_description,
            "reasoning_modules": reasoning_modules_str,
            "reasoning_formats": reasoning_formats
        },
        config={"callbacks": [langfuse_handler]}
    )