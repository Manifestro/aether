from typing import Iterable

from aether.domain.events import ToolCall, ToolResult
from aether.model.protocols import ToolExecutor


class AllowlistToolExecutor:
    """Rejects any tool call outside an explicit allowlist.

    Wraps a real ToolExecutor so that a Planner inventing a tool name (or
    naming one that exists but was not granted for this turn) gets a
    `tool_error` result instead of the call silently reaching the inner
    executor. This is the runtime-side enforcement counterpart to telling
    the Planner which tools it may use in the system prompt: prompting
    reduces how often the model invents a tool, this guarantees an invented
    or out-of-scope one never produces a spoken fact.
    """

    def __init__(self, allowed_tools: Iterable[str], inner: ToolExecutor) -> None:
        self._allowed_tools = frozenset(allowed_tools)
        self._inner = inner

    async def execute(self, call: ToolCall) -> ToolResult:
        if call.name not in self._allowed_tools:
            return ToolResult(call.call_id, call.name, {}, error=f"tool not allowed: {call.name}")
        return await self._inner.execute(call)
