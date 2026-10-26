import unittest
from app.harness.research_policy import Attempt, ResearchRecoveryPolicy, find_conflicts
from app.harness.schemas import ErrorCode, ToolError, ToolExecutionResult
from tests.helpers import evidence, success


class RecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_triggers_actual_compatible_fallback(self):
        calls = []
        async def invoke(source):
            calls.append(source)
            if source == "tavily":
                failure = ToolExecutionResult(source=source, tool_name="search", call_id="c", status="degraded", error=ToolError(code=ErrorCode.TIMEOUT, message="offline"))
                return Attempt(source, [failure])
            return Attempt(source, [success(source)])
        policy = ResearchRecoveryPolicy()
        outcome = await policy.collect("public technology", ["tavily"], invoke,
            allowed_sources=policy.compatible_sources("public technology", "tavily"))
        self.assertEqual(calls, ["tavily", "ragflow"])
        self.assertTrue(outcome.completed)
        self.assertEqual(outcome.failures[0]["code"], "TimeoutError")

    async def test_private_facts_are_not_replaced_with_web(self):
        calls = []
        async def invoke(source):
            calls.append(source)
            return Attempt(source)
        policy = ResearchRecoveryPolicy()
        allowed = policy.compatible_sources("查询内部库存", "mysql")
        result = await policy.collect("查询内部库存", ["mysql"], invoke, allowed_sources=allowed)
        self.assertFalse(result.completed)
        self.assertEqual(calls, ["mysql"])
        self.assertEqual(result.metadata()["outcome"], "aborted")

    async def test_two_source_loss_safely_aborts(self):
        async def invoke(source):
            return Attempt(source, [success(source)]) if source == "mysql" else Attempt(source)
        outcome = await ResearchRecoveryPolicy().collect("multi source", ["tavily", "mysql", "ragflow"], invoke, minimum_sources=2)
        self.assertFalse(outcome.completed)

    async def test_no_duplicate_recovery_and_stop_when_complete(self):
        calls = []
        async def invoke(source):
            calls.append(source)
            return Attempt(source, [success(source)])
        await ResearchRecoveryPolicy().collect("q", ["tavily"], invoke, allowed_sources=["tavily", "tavily", "ragflow"])
        self.assertEqual(calls, ["tavily"])

    def test_conflicts_are_flagged_not_resolved(self):
        items = [evidence(text="Price is 20 USD in 2025.", id="a"), evidence("ragflow", text="Price is 30 USD in 2025.", id="b")]
        result = find_conflicts(items)
        self.assertEqual(result[0]["evidence_ids"], ["a", "b"])
        self.assertIn("unresolved", result[0]["status"])
