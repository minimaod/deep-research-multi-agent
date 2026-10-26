import asyncio
import threading
import time
import unittest
from unittest.mock import Mock
from requests.exceptions import ConnectionError as RequestConnectionError, ReadTimeout
from app.harness.runtime import RunState, run_context
from app.harness.schemas import EmptyInput, ErrorCode, SearchInput
from app.harness.tool_wrapper import RetryPolicy, ToolFailure, ToolWrapper
from tests.helpers import payload


class ToolWrapperTests(unittest.TestCase):
    def wrapper(self, operation, **policy):
        return ToolWrapper("tavily", "internet_search", SearchInput, operation,
                           policy=RetryPolicy(delay_s=0, **policy), emit=Mock())

    def test_success_records_timing_and_events(self):
        wrapper = self.wrapper(Mock(return_value=payload()))
        result = wrapper.execute({"query": "test"})
        self.assertEqual(result.status, "ok")
        self.assertGreaterEqual(result.cost_ms, 0)
        self.assertEqual([c.args[0] for c in wrapper.emit.call_args_list], ["tool_start", "tool_end"])

    def test_timeout_retries_once_then_degrades(self):
        operation = Mock(side_effect=ReadTimeout("SECRET must not leak"))
        wrapper = self.wrapper(operation)
        result = wrapper.execute({"query": "test"})
        self.assertEqual(operation.call_count, 2)
        self.assertEqual(result.error.code, ErrorCode.TIMEOUT)
        self.assertNotIn("SECRET", result.model_dump_json())
        self.assertIn("tool_degraded", [c.args[0] for c in wrapper.emit.call_args_list])

    def test_connection_recovery(self):
        operation = Mock(side_effect=[RequestConnectionError(), payload()])
        result = self.wrapper(operation).execute({"query": "test"})
        self.assertEqual((result.status, result.attempts), ("ok", 2))

    def test_no_retry_for_empty_invalid_or_configuration(self):
        for code in (ErrorCode.EMPTY, ErrorCode.INVALID, ErrorCode.UNAVAILABLE):
            with self.subTest(code=code):
                op = Mock(side_effect=ToolFailure(code, "known_failure"))
                result = self.wrapper(op).execute({"query": "test"})
                self.assertEqual(op.call_count, 1)
                self.assertEqual(result.error.code, code)

    def test_schema_rejection_never_calls_provider(self):
        op = Mock()
        result = self.wrapper(op).execute({"query": "test", "max_results": 999})
        self.assertEqual(result.error.code, ErrorCode.INVALID)
        self.assertEqual(result.attempts, 0)
        op.assert_not_called()

    def test_bad_output_is_not_success(self):
        result = self.wrapper(Mock(return_value=({"kind": "wrong"}, [], []))).execute({"query": "test"})
        self.assertEqual(result.error.reason, "invalid_provider_output")

    def test_expired_sync_deadline_does_not_retry(self):
        def slow(*_):
            time.sleep(.02)
            return payload()
        result = self.wrapper(Mock(side_effect=slow), timeout_s=.005).execute({"query": "test"})
        self.assertEqual(result.error.code, ErrorCode.TIMEOUT)
        self.assertEqual(result.attempts, 1)

    def test_cancelled_error_propagates(self):
        with self.assertRaises(asyncio.CancelledError):
            self.wrapper(Mock(side_effect=asyncio.CancelledError())).execute({"query": "test"})


class AsyncWrapperTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_timeout_has_no_late_success(self):
        events = Mock()
        gate = threading.Event()
        def slow(*_):
            gate.wait(.2)
            return payload()
        state = RunState("test")
        token = run_context.set(state)
        try:
            wrapper = ToolWrapper("tavily", "search", EmptyInput, slow, RetryPolicy(timeout_s=.01), events)
            result = await wrapper.aexecute({})
            self.assertEqual(result.error.code, ErrorCode.TIMEOUT)
            self.assertEqual(state.workers, 1)
            gate.set()
            await state.wait_idle()
            await asyncio.sleep(0)
            self.assertEqual(len(state.results), 1)
            self.assertEqual([c.args[0] for c in events.call_args_list].count("tool_end"), 1)
            self.assertEqual({c.args[2]["call_id"] for c in events.call_args_list}, {result.call_id})
        finally:
            gate.set()
            run_context.reset(token)
