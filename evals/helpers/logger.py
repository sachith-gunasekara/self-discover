import logging

from pyprojroot import here

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(here("evals/logs/evaluation.log")),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger("self_discover_evals_logger")
