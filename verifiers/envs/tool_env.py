import json
from typing import Callable, cast

import verifiers as vf
from verifiers.types import AssistantMessage, Messages, ToolCall, ToolMessage
from verifiers.utils.async_utils import maybe_await
from verifiers.utils.tool_utils import (
    convert_func_to_tool_def,
    is_valid_tool_content_parts,
)


class ToolMonitorRubric(vf.Rubric):
    def __init__(self, tools: list[Callable] | None = None, **kwargs):
        super().__init__(**kwargs)

        self.tools = tools or []
        self.tool_names = [tool.__name__ for tool in self.tools]  # type: ignore[union-attr]

        # add tool metrics
        self.add_metric(self.total_tool_calls)
        for tool_name in self.tool_names:
            self.add_metric(self.get_tool_call_count_func(tool_name))

    def add_tool_metric(self, tool: Callable):
        pass

    def remove_tool_metric(self, tool: Callable):
        pass

    async def total_tool_calls(self, completion: Messages) -> float:
        """Count the total number of tool calls."""
        pass

    def get_tool_call_count_func(self, tool_name: str) -> Callable:
        """Create a metric that counts calls to a specific tool."""
        pass


class ToolEnv(vf.MultiTurnEnv):
    def __init__(
        self,
        tools: list[Callable] | None = None,
        max_turns: int = 10,
        error_formatter: Callable[[Exception], str] = lambda e: f"{e}",
        stop_errors: list[type[Exception]] | None = None,
        **kwargs,
    ):
        self.tools = tools or []
        self.max_turns = max_turns
        self.error_formatter = error_formatter
        self.stop_errors: list[type[Exception]] = stop_errors or []
        self.tool_defs = [convert_func_to_tool_def(tool) for tool in self.tools]
        self.tool_map = {
            getattr(tool, "__name__", tool.__class__.__name__): tool
            for tool in self.tools
        }
        super().__init__(tool_defs=self.tool_defs, max_turns=max_turns, **kwargs)

        self.tool_monitor_rubric = ToolMonitorRubric(tools=self.tools)
        self.add_rubric(self.tool_monitor_rubric)

    def _should_stop_for_error(self, err: Exception) -> bool:
        """Check if error is in stop_errors."""
        return any(isinstance(err, err_type) for err_type in self.stop_errors)

    def add_tool(self, tool: Callable):
        pass

    def remove_tool(self, tool: Callable):
        pass

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        pass

    async def call_tool(
        self, tool_name: str, tool_args: dict, tool_call_id: str, **kwargs
    ) -> ToolMessage:
        """Call a tool based on JSON command."""
        tool_func = self.tool_map[tool_name]
        result = await maybe_await(tool_func, **tool_args)
        content = result if is_valid_tool_content_parts(result) else str(result)
        return ToolMessage(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
        )

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs
    ) -> vf.Messages:
        last_msg = cast(vf.AssistantMessage, messages[-1])
        assert last_msg.tool_calls is not None
        tool_messages = []
        for tool_call in last_msg.tool_calls:
            tool_call_id: str = tool_call.id
            try:
                tool_name: str = tool_call.name
                tool_args: dict = json.loads(tool_call.arguments)
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
                continue  # skip tool call below

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
