import unittest

from aether.domain.events import ToolCall, ToolResult
from aether.runtime.tool_executor import AllowlistToolExecutor


class _RecordingExecutor:
    def __init__(self) -> None:
        self.calls = []

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call.call_id, call.name, {"ok": True})


class AllowlistToolExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_allowed_tool_reaches_the_inner_executor(self) -> None:
        inner = _RecordingExecutor()
        executor = AllowlistToolExecutor(["weather"], inner)
        call = ToolCall("c1", "weather", {})

        result = await executor.execute(call)

        self.assertTrue(result.succeeded)
        self.assertEqual(len(inner.calls), 1)

    async def test_unlisted_tool_is_rejected_without_reaching_inner_executor(self) -> None:
        inner = _RecordingExecutor()
        executor = AllowlistToolExecutor(["weather"], inner)
        call = ToolCall("c1", "chat", {})

        result = await executor.execute(call)

        self.assertFalse(result.succeeded)
        self.assertIn("chat", result.error)
        self.assertEqual(inner.calls, [])

    async def test_empty_allowlist_rejects_every_tool(self) -> None:
        inner = _RecordingExecutor()
        executor = AllowlistToolExecutor([], inner)

        result = await executor.execute(ToolCall("c1", "weather", {}))

        self.assertFalse(result.succeeded)
        self.assertEqual(inner.calls, [])


if __name__ == "__main__":
    unittest.main()
