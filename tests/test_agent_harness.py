import dataclasses
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command
from app.harness.middleware import ContextUsageMiddleware, DelegationHarnessMiddleware, ProviderHarnessMiddleware, combine_task_updates, guard_session_files, replace_message
from app.harness.research_policy import Attempt
from app.harness.runtime import RunState, run_context
from app.harness.token_counter import counter
from app.tools.tavily_tool import internet_search, search_wrapper
from tests.helpers import ScriptedModel, payload


def call(name, args, id):
    return {"name": name, "args": args, "id": id, "type": "tool_call"}


class HarnessTests(unittest.IsolatedAsyncioTestCase):
    def test_actual_application_factory_compiles_without_credentials(self):
        import sys
        from app.agent import main_agent as application
        model = ScriptedModel(responses=[AIMessage(content="ready")])
        previous = application.main_agent
        application.main_agent = None
        try:
            with patch.dict(sys.modules, {"app.agent.llm": SimpleNamespace(model=model)}):
                compiled = application.get_main_agent()
            self.assertTrue(callable(compiled.astream))
            self.assertIs(application.get_main_agent(), compiled)
        finally:
            application.main_agent = previous

    def test_command_replacement_preserves_all_state_and_ids(self):
        message = ToolMessage("raw", tool_call_id="task1", id="message1", artifact={"original": True})
        command = Command(update={"messages": [message], "files": {"a.md": "kept"}, "custom": 3}, goto="next")
        changed = replace_message(command, "trimmed")
        self.assertEqual(changed.update["files"], command.update["files"])
        self.assertEqual(changed.update["custom"], 3)
        self.assertEqual(changed.goto, "next")
        self.assertEqual(changed.update["messages"][0].tool_call_id, "task1")
        self.assertEqual(changed.update["messages"][0].id, "message1")
        self.assertEqual(message.content, "raw")

    def test_recovery_merges_fallback_file_state(self):
        first = Command(update={"messages": [ToolMessage("raw", tool_call_id="c")], "files": {"first": "a"}})
        second = Command(update={"messages": [ToolMessage("fallback", tool_call_id="c")], "files": {"second": "b"}})
        merged = combine_task_updates([Attempt("tavily", value=first), Attempt("ragflow", value=second)])
        self.assertEqual(merged.update["files"], {"first": "a", "second": "b"})
        self.assertEqual(merged.update["messages"][0].content, "raw")

    def test_file_tool_path_guard_blocks_escape(self):
        import tempfile
        from app.api.context import set_session_context, set_thread_context, reset_session_context
        with tempfile.TemporaryDirectory() as directory:
            a, b = set_session_context(directory), set_thread_context("guard")
            try:
                request = SimpleNamespace(tool_call=call("read_file_content", {"filename": "../outside.txt"}, "c"))
                self.assertIn("outside_session_directory", guard_session_files(request).content)
                request = SimpleNamespace(tool_call=call("generate_markdown", {"filename": "report.md"}, "c"))
                self.assertIsNone(guard_session_files(request))
            finally:
                reset_session_context(a, b)

    async def test_real_graph_automatically_delegates_fallback(self):
        from langchain_core.runnables import RunnableLambda
        from app.harness.schemas import ToolExecutionResult, ToolError, ErrorCode
        from app.harness.runtime import record_result
        from tests.helpers import success
        seen = []
        async def failed(_):
            seen.append("tavily")
            raise TimeoutError("injected child failure")
        async def recovered(_):
            seen.append("ragflow")
            record_result(success("ragflow", "Public technology evidence."))
            return {"messages": [AIMessage(content="recovered")]}
        parent = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("task", {"subagent_type": "网络搜索助手", "description": "public technology"}, "fallback1")]), AIMessage(content="done")])
        agent = create_deep_agent(model=parent, middleware=[DelegationHarnessMiddleware()], subagents=[
            {"name": "网络搜索助手", "description": "web", "runnable": RunnableLambda(failed)},
            {"name": "RAGFlow助手", "description": "docs", "runnable": RunnableLambda(recovered)}])
        with patch("app.api.monitor.monitor._emit"):
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "research"}]})
        message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(seen, ["tavily", "ragflow"])
        self.assertEqual(json.loads(message.content)["status"], "ok")
        self.assertEqual(message.tool_call_id, "fallback1")
        self.assertIn("TimeoutError", message.content)

    async def test_real_deepagents_task_boundary_budget(self):
        child = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("internet_search", {"query": "critical revenue"}, "web1")]), AIMessage(content="Child finished")])
        parent = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("task", {"subagent_type": "网络搜索助手", "description": "critical revenue"}, "task1")]), AIMessage(content="Final report")])
        agent = create_deep_agent(model=parent, middleware=[ContextUsageMiddleware(), DelegationHarnessMiddleware()],
            subagents=[{"name": "网络搜索助手", "description": "public web", "system_prompt": "Find evidence", "model": child,
                        "tools": [internet_search], "middleware": [ProviderHarnessMiddleware()]}])
        state = RunState("offline")
        token = run_context.set(state)
        try:
            with patch.object(search_wrapper, "operation", return_value=payload(text="Background discussion. " * 160 + "Critical revenue is 42 million.")), patch("app.api.monitor.monitor._emit"):
                result = await agent.ainvoke({"messages": [{"role": "user", "content": "research"}]})
            messages = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id == "task1"]
            self.assertEqual(len(messages), 1)
            self.assertLessEqual(counter.count(messages[0].content), 1500)
            self.assertIn("42 million", messages[0].content)
            self.assertEqual(json.loads(messages[0].content)["status"], "ok")
            self.assertEqual(len(state.context_counts), 2)
            self.assertEqual(state.workers, 0)
        finally:
            run_context.reset(token)

    async def test_schema_error_through_real_toolnode_is_structured(self):
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[call("internet_search", {"query": "x", "max_results": 999}, "bad1")]), AIMessage(content="Handled")])
        from langchain.agents import create_agent
        agent = create_agent(model, tools=[internet_search], middleware=[ProviderHarnessMiddleware()])
        with patch.object(search_wrapper, "operation") as operation, patch("app.api.monitor.monitor._emit"):
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "test"}]})
        operation.assert_not_called()
        msg = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertEqual(json.loads(msg.content)["error"]["code"], "InvalidParam")

    async def test_parallel_subtasks_do_not_mix_ledgers(self):
        from app.harness.runtime import subtask_context, SubtaskState
        import asyncio
        async def worker(source):
            token = subtask_context.set(SubtaskState(source, source))
            try:
                await asyncio.sleep(0)
                return subtask_context.get().source
            finally:
                subtask_context.reset(token)
        self.assertEqual(await asyncio.gather(worker("mysql"), worker("ragflow")), ["mysql", "ragflow"])
