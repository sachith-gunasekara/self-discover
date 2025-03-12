import os

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI

from .config import config
from self_discover._helpers.logger import logger


llama = True if "llama" in config["MODEL"]["model_type"] else False


# MODEL_ID = "hf:meta-llama/Llama-3.1-405B-Instruct" if llama else "mistral-large-2407" # Use if llama model is from glhf.chat
# MODEL_ID = "Meta-Llama-3.1-405B-Instruct" if llama else "mistral-large-2407" # Use if llama model is from Sambanovacloud
MODEL_ID = (
    "meta-llama/Meta-Llama-3.1-405B-Instruct" if llama else "mistral-large-2407"
)  # Use if llama model is from Hyperbolic

logger.info("Using {} for inference", MODEL_ID)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.083,
    check_every_n_seconds=5,
    max_bucket_size=1,
)

model_kwargs = {
    "temperature": 0.2,
    "top_p": 0.9,
    # "top_k": 15,
    "max_tokens": 8192,
}


if llama:
    # model = ChatSambaNovaCloud(
    #     model=MODEL_ID, rate_limiter=rate_limiter, **model_kwargs
    # )

    # model = ChatOpenAI(
    #     model=MODEL_ID,
    #     base_url="https://glhf.chat/api/openai/v1",
    #     rate_limiter=rate_limiter,
    #     **model_kwargs
    # )

    model = ChatOpenAI(
        model=MODEL_ID,
        base_url="https://api.hyperbolic.xyz/v1",
        api_key=os.environ["HYPERBOLIC_API_KEY"],
        rate_limiter=rate_limiter,
        **model_kwargs
    )

    # model = ChatNVIDIA(
    #     model=MODEL_ID,
    #     rate_limiter=rate_limiter,
    #     **model_kwargs
    # )
else:
    model = ChatMistralAI(model=MODEL_ID, rate_limiter=rate_limiter, **model_kwargs)
