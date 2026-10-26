import asyncio
import contextlib
import io
import json
import unittest
from collections import Counter
from eval.run_deep_research_benchmark import DATASET, grade_case, load_dataset, offline_network, render_report, run_case, run_suite


class BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with offline_network(), contextlib.redirect_stdout(io.StringIO()):
            cls.results = asyncio.run(run_suite())

    def test_dataset_split_and_fault_distribution(self):
        cases = load_dataset()
        self.assertEqual(Counter(c["category"] for c in cases), {"normal_multi_source": 12, "single_source_fault_injection": 7, "conflict_and_out_of_scope": 5})
        self.assertEqual(len({c["id"] for c in cases}), 24)
        profiles = Counter(c["context_workload"]["profile"] for c in cases if c["category"] == "normal_multi_source")
        self.assertEqual(profiles, {"compact": 3, "standard": 6, "long": 3})

    def test_all_case_behavior_assertions(self):
        self.assertEqual(self.results["passed"], 24)
        self.assertTrue(all(row["passed"] for row in self.results["cases"]))

    def test_fault_ablation_is_computed_and_double_failure_attributed(self):
        summary = self.results["summary"]
        self.assertEqual(summary["wrapper_success"], 6)
        self.assertEqual(summary["current_tool_baseline_success"], 4)
        self.assertEqual(summary["unprotected_success"], 0)
        recovered = next(row for row in self.results["experiment_a"] if row["id"] == "F01")
        path = [event["assistant_name"] for event in recovered["actual_delegations"] if event["call_id"] == "task-0-tavily"]
        self.assertEqual(path, ["网络搜索助手", "RAGFlow助手"])
        boundary = next(r for r in self.results["cases"] if r["id"] == "F07")
        self.assertFalse(boundary["completed"])
        self.assertEqual({e["source"] for e in boundary["failures"]}, {"tavily", "ragflow"})
        self.assertEqual({e["code"] for e in boundary["failures"]}, {"TimeoutError", "ServiceUnavailable"})

    def test_actual_graph_context_measurements_and_fact_retention(self):
        for row in self.results["experiment_b"]:
            with self.subTest(case=row["id"]):
                self.assertEqual(len(row["after"]["model_call_input_tokens_proxy"]), 3)
                self.assertLess(row["after"]["final_input_tokens_proxy"], row["before"]["final_input_tokens_proxy"])
                self.assertLess(row["after"]["effective_context_units"], row["before"]["effective_context_units"])
                self.assertEqual(row["after"]["effective_context_units"], sum(m["units"] for m in row["after"]["injected_messages"]))
                from app.harness.token_counter import counter
                for contribution in row["after"]["injected_messages"]:
                    actual = counter.count(json.dumps(contribution["payload"], ensure_ascii=False, separators=(",", ":")))
                    self.assertEqual(actual, contribution["units"])
                self.assertEqual(len(row["after"]["subtask_sizes"]), 5)
                self.assertTrue(all(size <= 1500 for size in row["after"]["subtask_sizes"]))
                self.assertTrue(all(o["units"] <= 6000 for o in row["after"]["provider_observations"]))
                self.assertTrue(row["after"]["required_fact_retained"])
        from eval.run_deep_research_benchmark import injected_message_volume
        from langchain_core.messages import HumanMessage, ToolMessage
        user = HumanMessage(content="query", id="user-1")
        evidence = ToolMessage(content="fact", tool_call_id="t1", id="tool-1")
        once = injected_message_volume([[user], [user, evidence]])
        self.assertEqual(once, injected_message_volume([[user], [user, evidence], [user, evidence]]))
        self.assertGreater(self.results["summary"]["reduction_percent"], 0)

    def test_grader_is_sensitive_to_deleted_or_changed_facts(self):
        case = next(c for c in load_dataset() if c["id"] == "C01")
        result = dict(next(r for r in self.results["cases"] if r["id"] == "C01"))
        self.assertTrue(grade_case(case, result))
        result["texts"] = result["texts"].replace("30 USD", "300 USD")
        self.assertFalse(grade_case(case, result))

    def test_expected_labels_do_not_control_execution(self):
        case = json.loads(json.dumps(load_dataset()[0]))
        case["expected"]["behavior"] = "abort"
        with offline_network():
            result = asyncio.run(run_case(case))
        self.assertEqual(result["behavior"], "complete")
        self.assertFalse(grade_case(case, result))

    def test_report_discloses_measurement_limits(self):
        report = render_report(self.results)
        self.assertIn("计量口径", report)
        self.assertIn("固定系统提示词", report)
        self.assertIn("阈值规则", report)
        self.assertIn("合成压力样本", report)
        self.assertIn("计数器", report)
