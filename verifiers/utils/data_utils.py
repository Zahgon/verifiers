# NOTE: Helper functions for example datasets. Not intended for core functionality.

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any, Callable, cast

from verifiers.types import Messages

if TYPE_CHECKING:
    from datasets import Dataset

### PROMPTS ###

THINK_BOXED_SYSTEM_PROMPT = "Think step-by-step inside <think>...</think> tags. \
    Then, give your final answer inside \\boxed{}."

### https://github.com/huggingface/lighteval/blob/ecef2c662b9418866b6447d33b5e7d5dedd74af8/src/lighteval/tasks/default_prompts.py#L1474
BOXED_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
###############


def format_dataset(
    dataset: Dataset,
    system_prompt: str | None = None,
    few_shot: Messages | None = None,
    question_key: str = "question",
    answer_key: str = "answer",
    map_kwargs: dict = {},
) -> Dataset:
    """
    Create `example_id` and `prompt` columns if not present.
    """
    pass


def extract_boxed_answer(text: str, strict: bool = False) -> str:
    """Extract the last \\boxed{...} answer from text.

    Args:
        text: The text to extract from.
        strict: If True, return "" when no \\boxed{} is found (for reward
            scoring where format compliance matters). If False, return the
            original text as a passthrough (for environments that use this
            as a general text extractor).
    """

    def find_matching_brace(s: str, start: int) -> int:
        pass

    # Find last \boxed{
    boxed_start = text.rfind("\\boxed{")
    if boxed_start == -1:
        return "" if strict else text
    # Find the content between the braces
    content_start = boxed_start + 7  # len('\\boxed{')
    closing_brace = find_matching_brace(text, content_start)

    if closing_brace == -1:
        return "" if strict else text

    return text[content_start:closing_brace]


def strip_non_numeric(text: str) -> str:
    pass


def extract_hash_answer(text: str) -> str:
    pass


def get_preprocess_fn(name: str) -> Callable[[dict], dict]:
    pass


def load_example_dataset(
    name: str = "gsm8k", split: str | None = None, n: int | None = None, seed: int = 0
) -> Dataset:
    pass
