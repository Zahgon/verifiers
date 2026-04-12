from collections.abc import Mapping
from typing import Any, Optional, cast

from openai import AsyncOpenAI, BaseModel
from openai.types.chat import ChatCompletion, ChatCompletionAssistantMessageParam

from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
    OpenAIChatMessage,
    OpenAIChatMessages,
    OpenAIChatResponse,
    OpenAITool,
    handle_openai_overlong_prompt,
)
from verifiers.types import SamplingArgs, State


def _has_multimodal_content(messages) -> bool:
    """Check if any message contains multimodal content (images, audio).

    Works with both plain dicts (OpenAIChatMessages) and Pydantic models
    (Messages stored in trajectory steps) since both support .get().
    """
    for msg in messages:
        content = msg.get("content") if hasattr(msg, "get") else None
        if isinstance(content, list):
            for part in content:
                if hasattr(part, "get") and part.get("type") in (
                    "image_url",
                    "input_audio",
                ):
                    return True
    return False


def _get_role(msg) -> str | None:
    pass


def _is_valid_env_tail(messages: list) -> bool:
    """Validate that messages follow env response patterns:
    all tool messages, with optionally a single user message last."""
    pass


# copy from vllm/entrypoints/openai/protocol.py
class TokenizeResponse(BaseModel):
    count: int
    max_model_len: int
    tokens: list[int]
    token_strs: Optional[list[str]] = None


class OpenAIChatCompletionsTokenClient(OpenAIChatCompletionsClient):
    """Wrapper for custom vLLM route /v1/chat/completions/tokens via AsyncOpenAI client."""

    @property
    def token_client(self) -> AsyncOpenAI:
        """Strips trailing /v1 from the OpenAI client."""
        pass

    @handle_openai_overlong_prompt
    async def get_native_response(
        self,
        prompt: OpenAIChatMessages,
        model: str,
        sampling_args: SamplingArgs,
        tools: list[OpenAITool] | None = None,
        **kwargs,
    ) -> OpenAIChatResponse:
        def normalize_sampling_args(sampling_args: SamplingArgs):
            sampling_args = dict(sampling_args)
            if "max_tokens" in sampling_args:
                sampling_args["max_completion_tokens"] = sampling_args.pop("max_tokens")
            sampling_args["logprobs"] = True
            extra_body = dict(return_token_ids=True)
            if "extra_body" in sampling_args:
                sampling_args["extra_body"] = {
                    **sampling_args["extra_body"],
                    **extra_body,
                }
            else:
                sampling_args["extra_body"] = extra_body
            return {k: v for k, v in sampling_args.items() if v is not None}

        sampling_args = normalize_sampling_args(sampling_args)
        state = cast(State, kwargs.pop("state"))
        extra_headers = kwargs.pop("extra_headers", None)
        # Use standard /chat/completions for: (1) first turn (no prior tokens
        # to stitch), or (2) conversations that contain multimodal content in
        # any turn.  vLLM ≤0.16's /tokenize doesn't run the multimodal
        # processor, so image placeholders stay collapsed (1 token instead of
        # N) and token-stitching (TITO) produces broken prompts.  Falling back
        # to message-based inference (MITO) lets vLLM handle expansion
        # correctly on every turn.
        has_multimodal = _has_multimodal_content(prompt) or any(
            _has_multimodal_content(step["prompt"]) for step in state["trajectory"]
        )
        if len(state["trajectory"]) == 0 or has_multimodal:
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )
        prompt_ids = await self.get_prompt_ids(state, prompt, tools)
        if prompt_ids is None:
            return await super().get_native_response(
                prompt, model, sampling_args, tools, extra_headers=extra_headers
            )

        extra_body = sampling_args.pop("extra_body", {})
        body = dict(
            model=model,
            messages=prompt,
            tools=tools,
            tokens=prompt_ids,
            **sampling_args,
            **extra_body,
        )

        return await self.client.post(
            "/chat/completions/tokens",
            body=body,
            cast_to=ChatCompletion,
            options={"headers": extra_headers} if extra_headers else {},
        )

    async def get_prompt_ids(
        self,
        state: State,
        prompt_messages: OpenAIChatMessages,
        oai_tools: list[OpenAITool] | None,
    ) -> list[int] | None:
        """
        Build prompt_ids for the next turn by stitching engine tokens with
        bridge tokens for the environment response.

        The engine's prev_turn_ids are preserved exactly (no retokenization),
        guaranteeing KV cache reuse via vLLM's prefix caching. Only the bridge
        tokens (env response + generation prompt) are new.

        Returns None to fall back to MITO when stitching is not possible.
        """

        def normalize_for_comparison(value: Any) -> Any:
            pass

        async def find_largest_prefix_match() -> tuple[list[int], bool, int] | None:
            """Scan trajectory backwards for the step whose messages form the
            longest prefix of prompt_messages. Returns
            (token_ids, is_truncated, prefix_len) or None."""
            pass

        match = await find_largest_prefix_match()
        if match is None:
            return None

        prev_turn_ids, is_truncated, prefix_len = match

        # Truncated completions have no stop token — can't reliably stitch.
        if is_truncated:
            self.logger.debug("TITO: truncated completion, falling back to MITO")
            return None

        # The env messages are everything after the prefix match.
        env_messages: OpenAIChatMessages = list(prompt_messages[prefix_len:])
        if not _is_valid_env_tail(env_messages):
            return None

        # Extract the bridge tokens using a minimal dual-tokenization that
        # avoids the problematic assistant message entirely. We tokenize:
        #   (a) [dummy_assistant, env_messages...]  with gen=True
        #   (b) [dummy_assistant]                   with gen=False
        # The bridge = (a)[cut_point:] where cut_point accounts for the gap
        # between the engine's stop token and the template's inter-turn separator.
        #
        # Using a dummy assistant message ensures the inter-turn separator between
        # assistant and env response is correct, while avoiding template behaviors
        # that depend on the assistant being the last message (e.g., Qwen3's
        # context-dependent think block injection with add_generation_prompt=False).
        dummy_assistant: OpenAIChatMessage = ChatCompletionAssistantMessageParam(
            role="assistant", content="x"
        )

        try:
            bridge_full_ids = await self.tokenize(
                messages=[dummy_assistant] + env_messages,
                tools=oai_tools,
                model=state["model"],
            )
            bridge_base_ids = await self.tokenize(
                messages=[dummy_assistant],
                tools=oai_tools,
                model=state["model"],
                extra_kwargs=dict(add_generation_prompt=False),
            )
        except Exception:
            self.logger.debug("TITO: bridge tokenization failed, falling back to MITO")
            return None

        # Verify the base is a prefix of the full (sanity check)
        if bridge_full_ids[: len(bridge_base_ids)] != bridge_base_ids:
            self.logger.debug(
                "TITO: bridge prefix property broken, falling back to MITO"
            )
            return None

        # The base ends at the template-rendered stop token + inter-turn separator.
        # The engine's prev_turn_ids ends at just the stop token.
        # The gap = tokens the template adds after the stop token (e.g., \n for Qwen).
        # We include the gap in the bridge so it covers everything after the stop token.
        #
        # Find the gap by locating the stop token in bridge_base_ids.
        # The stop token is the last completion_ids token from the matched step.
        stop_token_id = prev_turn_ids[-1]
        gap = 0
        for i in range(len(bridge_base_ids) - 1, -1, -1):
            if bridge_base_ids[i] == stop_token_id:
                gap = len(bridge_base_ids) - i - 1
                break

        bridge_ids = bridge_full_ids[len(bridge_base_ids) - gap :]

        # Handle stop tokens that double as role markers (e.g., GLM's <|observation|>):
        # if the bridge starts with the stop token that's already at the end of
        # prev_turn_ids, skip it to avoid duplication.
        if bridge_ids and bridge_ids[0] == stop_token_id:
            bridge_ids = bridge_ids[1:]

        return prev_turn_ids + list(bridge_ids)

    async def tokenize(
        self,
        messages: str | OpenAIChatMessages,
        tools: list[OpenAITool] | None,
        model: str,
        extra_kwargs: dict = {},
        **kwargs,
    ) -> list[int]:
        """Tokenize messages using the vLLM /tokenize API."""
        pass
