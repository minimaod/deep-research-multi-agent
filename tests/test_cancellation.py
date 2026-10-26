import asyncio
import threading
import unittest
from unittest.mock import Mock, patch
from fastapi import HTTPException
from app.api import server
from app.agent import main_agent
from app.harness.runtime import RunState, accept_event, active_runs, run_context
from app.harness.schemas import EmptyInput
from app.harness.tool_wrapper import RetryPolicy, ToolWrapper
from tests.helpers import payload


class CancellationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        server.active_tasks.clear()
        active_runs.clear()

    async def asyncTearDown(self):
        for task in list(server.active_tasks.values()):
            if not task.done():
                task.cancel()
        if server.active_tasks:
            await asyncio.gather(*server.active_tasks.values(), return_exceptions=True)
        server.active_tasks.clear()
        active_runs.clear()

    async def test_same_session_replacement_waits_for_old_exit(self):
        order = []
        async def fake(query, thread_id, run_id=None):
            order.append("start:" + query)
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(.01)
                order.append("exit:" + query)
        with patch.object(server, "run_deep_agent", side_effect=fake):
            first = await server.run_task(server.TaskRequest(query="one", thread_id="same"))
            await asyncio.sleep(0)
            second = await server.run_task(server.TaskRequest(query="two", thread_id="same"))
            await asyncio.sleep(0)
            self.assertLess(order.index("exit:one"), order.index("start:two"))
            self.assertNotEqual(first["run_id"], second["run_id"])
            await server.cancel_task("same")

    async def test_parallel_sessions_are_independent(self):
        gates = {"a": asyncio.Event(), "b": asyncio.Event()}
        async def fake(query, thread_id, run_id=None):
            await gates[thread_id].wait()
        with patch.object(server, "run_deep_agent", side_effect=fake):
            await asyncio.gather(*(server.run_task(server.TaskRequest(query="x", thread_id=id)) for id in gates))
            self.assertEqual(set(server.active_tasks), {"a", "b"})
            await server.cancel_task("a")
            self.assertFalse(server.active_tasks["b"].done())
            gates["b"].set()
            await asyncio.sleep(0)

    async def test_cancel_does_not_swallow_or_retry(self):
        state = RunState("cancel")
        state.cancelled.set()
        token = run_context.set(state)
        try:
            operation = Mock()
            wrapper = ToolWrapper("tavily", "x", EmptyInput, operation, emit=Mock())
            with self.assertRaises(asyncio.CancelledError):
                await wrapper.aexecute({})
            operation.assert_not_called()
        finally:
            run_context.reset(token)

    def test_late_events_suppressed_and_cancel_terminal_allowed(self):
        state = RunState("late")
        active_runs["late"] = state
        token = run_context.set(state)
        try:
            state.cancelled.set()
            self.assertFalse(accept_event("task_result"))
            self.assertFalse(accept_event("tool_end"))
            self.assertTrue(accept_event("task_cancelled"))
            active_runs["late"] = RunState("late")
            self.assertFalse(accept_event("task_cancelled"))
        finally:
            run_context.reset(token)

    async def test_busy_cleanup_returns_conflict_without_new_run(self):
        gate = asyncio.Event()
        async def old():
            try:
                await asyncio.Event().wait()
            finally:
                await gate.wait()
        task = asyncio.create_task(old())
        server.active_tasks["busy"] = task
        active_runs["busy"] = RunState("busy")
        original = active_runs["busy"]
        await asyncio.sleep(0)
        try:
            with patch.object(server, "run_deep_agent") as start:
                with self.assertRaises(HTTPException) as error:
                    await server.run_task(server.TaskRequest(query="new", thread_id="busy"))
                self.assertEqual(error.exception.status_code, 409)
                start.assert_not_called()
                self.assertIs(active_runs["busy"], original)
        finally:
            gate.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_physical_worker_stays_occupied_until_return(self):
        gate, started = threading.Event(), threading.Event()
        def operation(*_):
            started.set()
            gate.wait(.5)
            return payload()
        state = RunState("physical")
        token = run_context.set(state)
        try:
            wrapper = ToolWrapper("tavily", "x", EmptyInput, operation, RetryPolicy(timeout_s=1), Mock())
            task = asyncio.create_task(wrapper.aexecute({}))
            while not started.is_set():
                await asyncio.sleep(.001)
            state.cancelled.set()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(state.workers, 1)
            gate.set()
            await state.wait_idle()
            self.assertEqual(state.results, [])
        finally:
            gate.set()
            run_context.reset(token)
