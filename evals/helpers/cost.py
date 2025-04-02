import os
import json

from transformers import AutoTokenizer
from langchain_core.callbacks import BaseCallbackHandler

from self_discover._helpers.logger import logger

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")


class TokenCountingCallback(BaseCallbackHandler):
    def __init__(self, rate_per_1M_tokens, cost_cap, cost_file, lock):
        self.rate_per_1M_tokens = rate_per_1M_tokens
        self.cost_cap = cost_cap
        self.cost_file = cost_file
        self.lock = lock

        if not os.path.exists(self.cost_file):
            with open(self.cost_file, "w") as f:
                json.dump({"running_cost": 0}, f)

    def reset_values(self):
        """Reset token counts and cost for a new LLM call."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_cost = 0

    def calculate_cost(self):
        return ((self.prompt_tokens + self.completion_tokens) / 1e6) * self.rate_per_1M_tokens

    def read_running_cost(self):
        with self.lock:
            with open(self.cost_file, "r") as f:
                data = json.load(f)
            return data["running_cost"]

    def update_running_cost(self):
        running_cost = self.read_running_cost() + self.calculate_cost()

        with self.lock:
            with open(self.cost_file, "w") as f:
                json.dump({"running_cost": running_cost}, f)

    def is_under_cost_cap(self):
        return self.read_running_cost() <= self.cost_cap

    def can_make_api_call(self):
        if not self.is_under_cost_cap():
            logger.warning("Cost cap reached! No further API calls allowed.")
            return False

        logger.info("Allowing API call!")
        return True

    def on_chat_model_start(self, serialized, messages, **kwargs):
        if not self.can_make_api_call():
            raise RuntimeError("Cost cap exceeded. LLM call aborted.")

        self.reset_values()

        for i, message in enumerate(messages[0]):
            self.prompt_tokens += len(tokenizer.encode(message.content))

    def on_llm_end(self, response, **kwargs):
        for i, completion in enumerate(response.generations[0]):
            self.completion_tokens += len(tokenizer.encode(completion.text))

        self.update_running_cost()