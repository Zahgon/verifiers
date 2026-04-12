import inspect
import json
from abc import abstractmethod
from typing import Callable, cast

import verifiers as vf
from verifiers.types import Tool, ToolMessage
from verifiers.utils.tool_utils import convert_func_to_tool_def


def filter_signature(func, args_to_skip):
    """Return a wrapper with filtered signature for schema generation.

    Does not mutate the original function.
    """
    pass


class StatefulToolEnv(vf.ToolEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{e}",
        stop_errors: list[type[Exception]] | None = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            max_turns=max_turns,
            error_formatter=error_formatter,
            stop_errors=stop_errors,
            **kwargs,
        )
        self.tools: list[Callable] = tools or []
        self.tool_defs: list[Tool] = [
            convert_func_to_tool_def(tool) for tool in self.tools
        ]
        self.tool_map: dict[str, Callable] = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.tools
        }
        self.skipped_args: dict[str, list[str]] = {}
        self.max_turns: int = max_turns

    def add_tool(self, tool: Callable, args_to_skip: list[str] = []):
        """Add a tool, optionally hiding arguments from the agent's view.

        Skipped args are removed from the schema shown to the agent but can be
        injected at call time via update_tool_args. If a skipped arg uses a $ref
        to a type in $defs, that definition is also removed to keep the schema clean.

        Assumes all non-skipped args use standard JSON types (no remaining $ref/$defs).
        """
        pass

    def remove_tool(self, tool: Callable):
        pass

    @abstractmethod
    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        """Update tool arguments and/or state (in-place) based on messages and state."""
        pass

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        assert isinstance(messages, list)
        last_msg = cast(vf.AssistantMessage, messages[-1])
        assert last_msg.tool_calls is not None
        tool_messages = []
        for tool_call in last_msg.tool_calls:
            tool_call_id = tool_call.id
            try:
                tool_name: str = tool_call.name
                parsed_args = json.loads(tool_call.arguments)
                if not isinstance(parsed_args, dict):
                    raise ValueError(
                        f"Expected tool arguments to be a dict, got {type(parsed_args).__name__}: {parsed_args}"
                    )
                tool_args: dict = parsed_args
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolParseError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            tool_args = self.update_tool_args(
                tool_name, tool_args, messages, state, **kwargs
            )
            try:
                tool_message = await self.call_tool(tool_name, tool_args, tool_call_id)
                tool_messages.append(tool_message)
            except Exception as e:
                if self._should_stop_for_error(e):
                    raise vf.ToolCallError from e
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(e),
                        tool_call_id=tool_call_id,
                    )
                )

        return tool_messages
