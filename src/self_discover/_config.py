from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

# from ._helpers.handlers import langfuse_handler


@dataclass
class LLM:
    model: BaseChatModel


@dataclass
class Langfuse:
    CONFIG = {
        "configurable": {"thread_id": 1}, 
        # "callbacks": [langfuse_handler]
    }