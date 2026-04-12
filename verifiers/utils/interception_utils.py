"""Utilities for intercepting API calls from agents running in sandboxes."""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, cast

from aiohttp import web
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)
from openai.types.chat.chat_completion_chunk import (
    Choice as ChunkChoice,
)

from verifiers.types import Response
from verifiers.utils.logging_utils import truncate

logger = logging.getLogger(__name__)


class InterceptionServer:
    """
    HTTP server that intercepts API requests from agents.

    Requests are queued for processing, and responses are delivered back
    to the agent once the actual model response is obtained.
    """

    def __init__(self, port: int):
        self.port = port
        self._app: Any = None
        self._runner: Any = None
        self._site: Any = None
        self._lock = asyncio.Lock()

        # Track active rollouts and their request queues
        self.active_rollouts: dict[str, dict[str, Any]] = {}
        # Track individual intercepts (request_id -> intercept data)
        self.intercepts: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        async with self._lock:
            if self._app is not None:
                return

            app = web.Application()
            app.router.add_post(
                "/rollout/{rollout_id}/v1/chat/completions",
                self._handle_request,
            )
            app.router.add_get(
                "/health",
                lambda _: web.json_response({"status": "ok"}),
            )

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", self.port)
            await site.start()

            self._app = app
            self._runner = runner
            self._site = site

            # OS-assigned port if port=0
            if self.port == 0:
                server = getattr(site, "_server", None)
                sockets = getattr(server, "sockets", None) if server else None
                if sockets:
                    self.port = sockets[0].getsockname()[1]
            if self.port == 0:
                raise RuntimeError("Failed to resolve OS-assigned port")

            logger.debug(f"Started interception server on port {self.port}")

    async def stop(self) -> None:
        async with self._lock:
            if self._runner is not None:
                try:
                    await self._runner.cleanup()
                    logger.debug("Stopped HTTP interception server")
                except RuntimeError as e:
                    if "Event loop is closed" not in str(e):
                        raise
                    logger.debug("HTTP server cleanup skipped (event loop closed)")
                finally:
                    self._runner = None
                    self._site = None
                    self._app = None

    def register_rollout(self, rollout_id: str) -> asyncio.Queue:
        request_queue: asyncio.Queue = asyncio.Queue()
        self.active_rollouts[rollout_id] = {
            "request_id_queue": request_queue,
        }
        return request_queue

    def unregister_rollout(self, rollout_id: str) -> None:
        # Cancel any pending intercepts for this rollout
        pass

    async def _handle_request(self, request: Any) -> Any:
        pass

    async def _handle_streaming_response(
        self, http_request: Any, rollout_id: str, intercept: dict
    ) -> Any:
        pass


def deliver_response(
    intercept: dict,
    response: Response | ChatCompletion | None,
    error: BaseException | None = None,
) -> None:
    future = intercept.get("response_future")
    if future and not future.done():
        if error is not None:
            future.set_exception(error)
        elif response is not None:
            future.set_result(response)


async def synthesize_stream(
    intercept: dict, response: Response | None, error: BaseException | None = None
) -> None:
    """Deliver a complete ChatCompletion as synthetic SSE chunks to the agent.

    Allows the base-class get_model_response (non-streaming, TITO-aware) to be
    used for the vLLM call while still satisfying agents that request streaming.

    Protocol (must match _handle_streaming_response):
      put chunk(s) on chunk_queue → put None (EOF) → resolve response_future.
    """
    chunk_queue = cast(
        asyncio.Queue[dict | None] | None,
        intercept.get("chunk_queue"),
    )
    future = cast(asyncio.Future[Any] | None, intercept.get("response_future"))

    # Error / no-response: unblock queue reader, fail/resolve future
    if error is not None or response is None:
        if chunk_queue is not None:
            try:
                chunk_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        if future and not future.done():
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(None)
        return

    if chunk_queue is None:
        raise RuntimeError("Missing chunk_queue for streaming interception")

    message = response.message

    # Chunk 1: content + tool_calls in delta
    delta_tool_calls = None
    if message.tool_calls:
        delta_tool_calls = [
            ChoiceDeltaToolCall(
                index=i,
                id=tc.id,
                type="function",
                function=ChoiceDeltaToolCallFunction(
                    name=tc.name,
                    arguments=tc.arguments,
                ),
            )
            for i, tc in enumerate(message.tool_calls)
        ]

    delta_content: str | None
    if isinstance(message.content, str):
        delta_content = message.content
    elif isinstance(message.content, list):
        text_parts: list[str] = []
        for part in message.content:
            text = (
                part.get("text")
                if isinstance(part, dict)
                else getattr(part, "text", None)
            )
            if isinstance(text, str):
                text_parts.append(text)
        delta_content = "".join(text_parts) if text_parts else None
    else:
        delta_content = None

    content_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(
                    role="assistant",
                    content=delta_content,
                    tool_calls=delta_tool_calls,
                ),
                finish_reason=None,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    content_chunk_dict = content_chunk.model_dump()
    if message.reasoning_content:
        content_chunk_dict["choices"][0]["delta"]["reasoning_content"] = (
            message.reasoning_content
        )
    await chunk_queue.put(content_chunk_dict)

    # Chunk 2: finish_reason only
    finish_chunk = ChatCompletionChunk(
        id=response.id,
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(),
                finish_reason=message.finish_reason,
            )
        ],
        created=response.created,
        model=response.model,
        object="chat.completion.chunk",
    )
    finish_chunk_dict = finish_chunk.model_dump()
    await chunk_queue.put(finish_chunk_dict)

    # EOF sentinel + resolve future
    await chunk_queue.put(None)
    if future and not future.done():
        future.set_result(response)


def create_empty_completion(model: str) -> ChatCompletion:
    pass


# Logging helpers


def _response_content_to_text(content: Any) -> str:
    pass


def serialize_intercept_response(response: Any) -> dict[str, Any]:
    """Serialize intercepted responses to OpenAI ChatCompletion JSON shape."""
    pass


def _log_request(rollout_id: str, body: dict) -> None:
    """Log an intercepted request."""
    pass


def _log_response(rollout_id: str, response: dict) -> None:
    """Log the response from the model."""
    pass
