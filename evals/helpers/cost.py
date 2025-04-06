import os
import json
import atexit

from transformers import AutoTokenizer
from langchain_core.callbacks import BaseCallbackHandler

from self_discover._helpers.logger import logger

from .kafka_producer import KafkaProducerHelper


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-405B-Instruct")


TARGET_TOPIC = "self-discover-cost"


class TokenCountingCallback(BaseCallbackHandler):
    def __init__(
        self, rate_per_1M_tokens, cost_cap, current_running_cost, producer_key
    ):
        self.rate_per_1M_tokens = rate_per_1M_tokens
        self.cost_cap = cost_cap
        self.running_cost = current_running_cost
        self.producer_key = producer_key
        self.pid = os.getpid()

        self.producer = None

        self.producer = KafkaProducerHelper(TARGET_TOPIC, self.producer_key)
        atexit.register(self._cleanup_producer)
        logger.info(
            "[PID {}] Kafka producer initialized for callback and registered for atexit cleanup.",
            self.pid,
        )

    def _cleanup_producer(self):
        logger.info("[PID {}] Attempting producer cleanup via atexit...", self.pid)
        if self.producer:
            try:
                self.producer.close()
                self.producer = None
            except Exception as e:
                logger.error("[PID {}] Error during producer cleanup: {}", self.pid, e)
        else:
            logger.info(
                "[PID {}] Producer already closed or not initialized during cleanup.",
                self.pid,
            )

    def reset_values(self):
        """Reset token counts and cost for a new LLM call."""
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def calculate_cost(self):
        cost = (
            (self.prompt_tokens + self.completion_tokens) / 1e6
        ) * self.rate_per_1M_tokens

        self.running_cost += cost

        return cost

    def update_running_cost(self):
        payload = self.producer.generate_event_payload(
            os.getpid(),
            self.prompt_tokens,
            self.completion_tokens,
            self.calculate_cost(),
        )

        self.producer.produce(payload)

    def is_under_cost_cap(self):
        return self.running_cost <= self.cost_cap

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
