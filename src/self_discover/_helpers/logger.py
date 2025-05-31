import os

from loguru import logger
from logtail import LogtailHandler
from pyprojroot import here


# handler = LogtailHandler(
#     source_token=os.getenv("BETTERSTACK_TELEMETRY_SOURCE_TOKEN"),
#     host=os.getenv("BETTERSTACK_TELEMETRY_INGESTING_HOST"),
# )


logger.add(here("evals/logs/evaluation.log"))
# logger.add(handler)
