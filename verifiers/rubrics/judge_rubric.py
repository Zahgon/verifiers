from typing import Any

from openai import APIError, APITimeoutError, AsyncOpenAI, RateLimitError

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, State
from verifiers.utils.async_utils import maybe_await

DEFAULT_JUDGE_PROMPT = """Given a ground truth answer \
and a response, determine if the response is correct.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only."""


class JudgeRubric(Rubric):
    def __init__(
        self,
        parser: Parser | None = None,
        parallelize_scoring: bool = False,
        judge_client: AsyncOpenAI | None = None,
        judge_model: str = "gpt-4.1-nano",
        judge_sampling_args: dict[str, Any] | None = None,
        judge_prompt: str = DEFAULT_JUDGE_PROMPT,
    ):
        super().__init__(parser=parser)
        self.judge_client = judge_client if judge_client is not None else AsyncOpenAI()
        self.judge_model = judge_model
        self.judge_prompt = judge_prompt
        self.judge_sampling_args = judge_sampling_args or {}
        self.class_objects = {
            "parser": self.parser,
            "judge": self.judge,
            "judge_client": self.judge_client,
            "judge_model": self.judge_model,
            "judge_prompt": self.judge_prompt,
            "judge_sampling_args": self.judge_sampling_args,
        }

    async def judge(
        self,
        prompt: Messages,
        completion: Messages,
        answer: str,
        state: State | None = None,
    ) -> str:
        pass
