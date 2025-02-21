import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from self_discover import validate_params, self_discover
from langchain_mistralai import ChatMistralAI
from langgraph.types import Send
from langchain_core.rate_limiters import InMemoryRateLimiter
from unittest.mock import patch

model_kwargs = {"temperature": 0.2, "top_p": 0.9, "top_k": 15, "max_tokens": 10240}

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.9,
    check_every_n_seconds=1,
    max_bucket_size=1,
)

task_descriptions = [
    r"Suppose that $4^{a}=5$, $5^{b}=6$, $6^{c}=7,$ and $7^{d}=8$. What is $a\cdot b\cdot c\cdot d$?",
    r"Find all values of $x$ that satisfy the equation $x = \!\sqrt{11-2x} + 4$.",
    r"Two positive numbers $p$ and $q$ have the property that their sum is equal to their product. If their difference is $7$, what is $\frac{1}{\frac{1}{p^2}+\frac{1}{q^2}}$? Your answer will be of the form $\frac{a+b\sqrt{c}}{d}$, where $a$ and $b$ don't both share the same common factor with $d$ and $c$ has no square as a factor. Find $a+b+c+d$.",
]
answer_formats = "The final value of the word problem without any expalainations."

model = ChatMistralAI(
    model="mistral-large-2407", rate_limiter=rate_limiter, **model_kwargs
)


def test_validate_params_valid_input():
    try:
        validate_params(task_descriptions, model, answer_formats)
    except ValueError:
        pytest.fail("validate_params raised ValueError unexpectedly!")


def test_validate_params_invalid_model():
    invalid_model = "invalid model"
    with pytest.raises(
        ValueError,
        match="The model must be an instance of BaseChatModel. Ensure you have defined your LangChain Chat Model properly.",
    ):
        validate_params(task_descriptions, invalid_model, answer_formats)


def test_validate_params_empty_task_descriptions():
    empty_task_descriptions = []
    with pytest.raises(
        ValueError, match="task_descriptions must be a non-empty list of strings."
    ):
        validate_params(empty_task_descriptions, model, answer_formats)


def test_validate_params_invalid_task_descriptions_type():
    invalid_task_descriptions = [123, 456]
    with pytest.raises(
        ValueError, match="task_descriptions must be a non-empty list of strings."
    ):
        validate_params(invalid_task_descriptions, model, answer_formats)


def test_validate_params_invalid_answer_formats():
    invalid_answer_formats = 12345
    with pytest.raises(ValueError, match="answer_formats must be a string."):
        validate_params(task_descriptions, model, invalid_answer_formats)


def test_validate_params_empty_answer_formats():
    empty_answer_formats = None
    with pytest.raises(ValueError, match="answer_formats must be a string."):
        validate_params(task_descriptions, model, empty_answer_formats)


def test_self_discover_default():
    result = self_discover(task_descriptions, model, answer_formats)
    assert isinstance(result, dict)


def test_self_discover_phase_1():
    result = self_discover(task_descriptions, model, answer_formats, phase=1)
    assert isinstance(result, dict)


def test_self_discover_phase_2():
    with patch(
        "self_discover.continue_to_reason",
        return_value=[
            Send(
                "reason",
                {
                    "task_description": "Test task description",
                    "answer_formats": "Test formats",
                    "reasoning_structure": '{"step-by-step reasoning": , "answer"}',
                },
            )
            for task_description in range(2)
        ],
    ):
        result = self_discover(task_descriptions, model, answer_formats, phase=2)
        assert isinstance(result, dict)


def test_self_discover_invalid_phase():
    with pytest.raises(ValueError, match="Invalid phase. Must be 0, 1, or 2."):
        self_discover(task_descriptions, model, answer_formats, phase=3)


def test_self_discover_stream_true():
    result = self_discover(task_descriptions, model, answer_formats, stream=True)
    assert isinstance(result, dict)
