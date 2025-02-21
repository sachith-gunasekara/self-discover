import pytest
from langchain_mistralai import ChatMistralAI
from langchain_core.rate_limiters import InMemoryRateLimiter
from self_discover import self_discover


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


def test_self_discover_valid_input():
    result = self_discover(task_descriptions, model, answer_formats)

    assert isinstance(result, dict)
    assert "selected_modules" in result
    assert "adapted_modules" in result
    assert "reasoning_structure" in result
    assert "reasonings" in result


def test_self_discover_invalid_model():
    model = "invalid model"

    with pytest.raises(ValueError, match="model must be an instance of BaseChatModel. Are you sure you defined your LanggChain Chat Model properly."):
        self_discover(task_descriptions, model, answer_formats)


def test_self_discover_empty_task_descriptions():
    task_descriptions = []
    
    with pytest.raises(
        ValueError, match="task_descriptions must be a non-empty list of strings."
    ):
        self_discover(task_descriptions, model, answer_formats)
