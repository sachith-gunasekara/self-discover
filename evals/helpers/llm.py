import os
import json
import threading

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from pyprojroot import here

from .config import config
from .cost import TokenCountingCallback
from self_discover._helpers.logger import logger


llama = True if "llama" in config["MODEL"]["model_type"] else False

RATE_PER_1M_TOKENS = 0.8
COST_CAP = 118.0  # USD
PRODUCER_KEY = "running_cost"
OUTPUT_FILE = here(f"evals/logs/running_cost.json")
MODEL_ID = (
    "llama3.1-405b-instruct-fp8" if llama else "mistral-large-2407"
)  # Use if llama model is from Hyperbolic

logger.info("Using {} for inference", MODEL_ID)


model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 8192,
}

lock = threading.Lock()
def read_running_cost():
    with lock:
        with open(OUTPUT_FILE, "r") as f:
            data = json.load(f)

    return data["running_cost"]

token_callback = TokenCountingCallback(RATE_PER_1M_TOKENS, COST_CAP, read_running_cost(), PRODUCER_KEY)

if llama:
    model = ChatOpenAI(
        model=MODEL_ID,
        base_url="https://api.lambdalabs.com/v1",
        api_key=os.environ["LAMBDALABS_API_KEY"],
        callbacks=[token_callback],
        **model_kwargs
    )
else:
    model_kwargs["top_k"] = 15
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1,
        check_every_n_seconds=5,
        max_bucket_size=1,
    )

    model = ChatMistralAI(model=MODEL_ID, rate_limiter=rate_limiter, **model_kwargs)
