from langchain_core.language_models.chat_models import BaseChatModel

from .logger import logger

def validate_params(
    task_description: list[str] | str, model: BaseChatModel, answer_formats: str
):
    logger.debug("Validating parameters")

    if not isinstance(model, BaseChatModel):
        logger.error("Invalid model type: {}", type(model))
        raise ValueError(
            "The model must be an instance of BaseChatModel. Ensure you have defined your LangChain Chat Model properly."
        )

    if isinstance(task_description, str):
        task_description = [task_description]

    if not task_description or not all(
        isinstance(task, str) for task in task_description
    ):
        logger.error("Invalid task_description: {}", task_description)
        raise ValueError("task_description must be a non-empty list of strings.")

    if not isinstance(answer_formats, str):
        logger.error("Invalid answer_formats type: {}", type(answer_formats))
        raise ValueError("answer_formats must be a string.")