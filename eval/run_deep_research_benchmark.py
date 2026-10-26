"""Run offline harness evals and real-graph context replay, without provider APIs.

Usage: python eval/run_deep_research_benchmark.py
The frozen fake model tests execution wiring, not the intelligence of an LLM.
"""
import argparse
import asyncio
import contextlib
import hashlib
import inspect
import io
import json
import socket
import subprocess
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from mysql.connector import Error as MySQLError
from requests.exceptions import ConnectionError as HTTPConnectionError, ReadTimeout

from app.agent.prompts import main_agent_content
from app.harness.context_trimmer import trim_context
from app.harness.middleware import ContextUsageMiddleware, DelegationHarnessMiddleware, ProviderHarnessMiddleware
from app.harness.research_policy import Attempt, ResearchOutcome, ResearchRecoveryPolicy, SOURCE_AGENTS
from app.harness.runtime import RunState, run_context
from app.harness.schemas import AnswerData, AssistantData, AssistantInfo, EmptyInput, ErrorCode, EvidenceItem, SearchData, SearchHit, SQLData
from app.harness.skill_backend import make_skill_backend
from app.harness.sql_guard import validate_readonly_sql
from app.harness.token_counter import counter
from app.harness.tool_wrapper import RetryPolicy, ToolFailure, ToolWrapper
from app.tools.markdown_tools import generate_markdown
from app.tools.pdf_tools import convert_md_to_pdf
from app.tools.upload_file_read_tool import read_file_content
from app.tools import db_tools, ragflow_tools, tavily_tool
from eval.baseline import BASELINE_COMMIT, evaluate_baseline

DATASET = ROOT / "eval" / "golden_dataset_24.json"
EFFECTIVE_CONTEXT_LIMIT = 12000


class ReplayModel(FakeMessagesListChatModel):
    seen_messages: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class ExpertReplayModel(BaseChatModel):
    """Frozen expert trajectory using actual registered tools and middleware."""
    source: str
    query: str
    seen_messages: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-expert-tool-replay"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        observed = [m for m in messages if isinstance(m, ToolMessage)]
        if self.source == "tavily":
            name, args = "internet_search", {"query": self.query}
        elif self.source == "mysql":
            name, args = ("list_sql_tables", {}) if not observed else ("get_table_data", {"table_name": "fixture_data"})
        else:
            name, args = ("get_assistant_list", {}) if not observed else ("create_ask_delete", {"chat_name": "fixture-research", "question": self.query})
        count = 1 if self.source == "tavily" else 2
        if len(observed) >= count:
            message = AIMessage(content="Evidence collection finished; retain source references and report gaps.")
        else:
            message = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"{self.source}-tool-{len(observed)}", "type": "tool_call"}])
        return ChatResult(generations=[ChatGeneration(message=message)])


def injected_message_volume(snapshots):
    """Count each new non-system input message once across a complete task.

    Message IDs identify repeated history; IDs themselves are excluded from the
    counted wire representation. Distinct tool returns still count separately,
    even when they contain the same evidence from repeated retrievals.
    """
    seen, contributions = set(), []
    for messages in snapshots:
        for position, message in enumerate(messages):
            if message.type == "system":
                continue
            key = message.id or f"{position}:{message.type}"
            if key in seen:
                continue
            seen.add(key)
            wire = {"role": message.type, "content": message.content}
            if getattr(message, "tool_calls", None):
                wire["tool_calls"] = message.tool_calls
            serialized = json.dumps(wire, ensure_ascii=False, separators=(",", ":"))
            contributions.append({"message_id": key, "role": message.type, "payload": wire, "units": counter.count(serialized),
                                  "sha256": hashlib.sha256(serialized.encode()).hexdigest()})
    return {"total_units": sum(row["units"] for row in contributions), "messages": contributions}


@contextlib.contextmanager
def offline_network():
    original = socket.socket.connect
    def guarded(sock, address):
        if any(frame.function == "_fallback_socketpair" for frame in inspect.stack()):
            return original(sock, address)
        raise AssertionError("External network forbidden in benchmark")
    with patch("socket.socket.connect", new=guarded), patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "LANGCHAIN_TRACING_V2": "false"}):
        yield


def load_dataset(path=DATASET):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = payload["cases"]
    if payload.get("schema_version") != 2:
        raise ValueError("benchmark dataset schema_version must be 2")
    expected = {"normal_multi_source": 12, "single_source_fault_injection": 7, "conflict_and_out_of_scope": 5}
    if Counter(case["category"] for case in cases) != expected or len({case["id"] for case in cases}) != 24:
        raise ValueError("dataset requires 24 unique cases in the 12/7/5 split")
    for case in cases:
        if not set(case["sources"]).issubset(SOURCE_AGENTS) or not 1 <= case["minimum_sources"] <= 3:
            raise ValueError("invalid source configuration")
        if any(code not in ("TimeoutError", "ServiceUnavailable") for code in case["faults"].values()):
            raise ValueError("invalid injected fault")
        for source, documents in case["documents"].items():
            for raw in documents:
                item = EvidenceItem.model_validate(raw)
                if item.source != source:
                    raise ValueError("evidence source mismatch")
    faults = [c for c in cases if c["category"] == "single_source_fault_injection"]
    if sorted(len(c["faults"]) for c in faults) != [1, 1, 1, 1, 1, 1, 2]:
        raise ValueError("six single faults and one double fault required")
    normal = [c for c in cases if c["category"] == "normal_multi_source"]
    profiles = Counter(c["context_workload"]["profile"] for c in normal)
    if profiles != {"compact": 3, "standard": 6, "long": 3}:
        raise ValueError("context workloads require the declared 3/6/3 profile split")
    if any(c["context_workload"]["phases"] != payload["context_stress_design"]["phases"] for c in normal):
        raise ValueError("context phases must match the declared dataset workload")
    if payload["context_stress_design"]["effective_context_limit"] != EFFECTIVE_CONTEXT_LIMIT:
        raise ValueError("default effective context limit must match the dataset")
    return cases


def fixture_operation(case, source):
    """Mock at the provider boundary, not the policy or scoring function."""
    def operation(*_):
        code = case["faults"].get(source)
        if code:
            if source == "mysql":
                raise MySQLError("injected fixture fault", errno=3024 if code == "TimeoutError" else 2003)
            raise ReadTimeout("injected fixture fault") if code == "TimeoutError" else HTTPConnectionError("injected fixture fault")
        items = [EvidenceItem.model_validate(item) for item in case["documents"].get(source, [])]
        if not items:
            raise ToolFailure(ErrorCode.EMPTY, "fixture_empty")
        if source == "tavily":
            data = SearchData(query=case["query"], results=[SearchHit(title=i.title, url=i.locator, content=i.text, score=i.score) for i in items])
        elif source == "mysql":
            data = SQLData(columns=["finding"], rows=[[i.text] for i in items])
        else:
            data = AnswerData(answer="\n".join(i.text for i in items), references=[i.locator for i in items], references_missing=False)
        return data, items, []
    return Mock(side_effect=operation)


async def run_case(case):
    # Expected labels are only consumed later by grade_case, never by this driver.
    if case.get("probe_sql"):
        try:
            validate_readonly_sql(case["probe_sql"])
            return {"behavior": "unsafe", "completed": False, "evidence": [], "failures": [], "texts": ""}
        except ToolFailure as error:
            return {"behavior": "reject", "completed": False, "evidence": [], "failures": [{"code": error.code.value}], "texts": ""}

    events = []
    async def invoke(source):
        wrapper = ToolWrapper(source, "fixture_research", EmptyInput, fixture_operation(case, source),
            RetryPolicy(timeout_s=1, delay_s=0), lambda *args: events.append(args))
        result = await wrapper.aexecute({})
        return Attempt(source, [result])
    outcome = await ResearchRecoveryPolicy().collect(case["query"], case["sources"], invoke,
        allowed_sources=case["compatible_sources"], minimum_sources=case["minimum_sources"])
    clipped = [trim_context(attempt.evidence, case["query"], 1500, {"source": attempt.source}) for attempt in outcome.attempts]
    retained = [e for item in clipped for e in item.evidence]
    return {"behavior": "complete" if outcome.completed else "abort", "completed": outcome.completed,
        "evidence": [e.model_dump(mode="json") for e in retained], "texts": "\n".join(e.text for e in retained),
        "failures": outcome.failures, "conflicts": outcome.metadata()["conflicts"],
        "tool_events": [{"event": event, **data} for event, _, data in events],
        "trim": [item.stats() for item in clipped]}


def grade_case(case, result):
    expected = case["expected"]
    if expected["behavior"] == "preserve_conflict":
        return result["completed"] and bool(result.get("conflicts")) and all(text in result["texts"] for text in expected["required_text"])
    return (result["behavior"] == expected["behavior"]
            and all(text in result["texts"] for text in expected["required_text"]))


async def replay_supervisor_context(case, trim_enabled):
    """Replay actual expert tools, both budget stages, and the task return path.

    Workload phases live in the dataset. Every changed setting is shared across
    the two groups except provider observation trimming and delegation trimming.
    """
    phases = case.get("context_workload", {}).get("phases", [case["sources"]])
    tools = {"tavily": [tavily_tool.internet_search], "mysql": [db_tools.list_sql_tables, db_tools.get_table_data],
             "ragflow": [ragflow_tools.get_assistant_list, ragflow_tools.create_ask_delete]}
    experts = {source: ExpertReplayModel(source=source, query=case["query"]) for source in case["sources"]}
    children = [{"name": SOURCE_AGENTS[source], "description": f"Fixture evidence from {source}",
        "system_prompt": "Collect source evidence using the registered tools, keep locators and disclose errors.",
        "model": experts[source], "tools": tools[source],
        "middleware": [ProviderHarnessMiddleware(trim_enabled=trim_enabled)]} for source in case["sources"]]
    responses = []
    for phase_index, sources in enumerate(phases):
        phase = "discovery" if phase_index == 0 else "verification"
        description = (f"{case['query']} Phase: {phase}. Return dated source evidence and distinguish facts, "
                       "limitations and conflicting definitions.")
        calls = [{"name": "task", "args": {"subagent_type": SOURCE_AGENTS[source], "description": description},
                  "id": f"task-{phase_index}-{source}", "type": "tool_call"} for source in sources]
        responses.append(AIMessage(content="", tool_calls=calls))
    responses.append(AIMessage(content="Fixture report: use the provided evidence."))
    model = ReplayModel(responses=responses)
    agent = create_deep_agent(model=model, system_prompt=main_agent_content["system_prompt"],
        tools=[generate_markdown, convert_md_to_pdf, read_file_content], subagents=children,
        middleware=[ContextUsageMiddleware(), DelegationHarnessMiddleware(trim_enabled=trim_enabled)],
        backend=make_skill_backend(), skills=["/skills/"])
    run = RunState("benchmark-" + case["id"])
    token = run_context.set(run)
    source_operations = {source: fixture_operation(case, source) for source in case["sources"]}

    def discovery_operation(source):
        def operation(params, timeout):
            # Apply the same outage to discovery and factual retrieval.
            if case["faults"].get(source):
                return source_operations[source](params, timeout)
            if source == "mysql":
                item = EvidenceItem(id="fixture-table", source="mysql", kind="row", locator="mysql:fixture:schema",
                    text='{"columns":["table"],"values":["fixture_data"]}')
                return SQLData(columns=["table"], rows=[["fixture_data"]]), [item], []
            return AssistantData(assistants=[AssistantInfo(name="fixture-research", description="Synthetic research documents", datasets=["fixture-data"])]), [], []
        return operation

    def evidence_operation(source):
        def operation(params, timeout):
            return source_operations[source](params, timeout)
        return operation

    wrappers = [(tavily_tool.search_wrapper, evidence_operation("tavily")),
                (db_tools.list_wrapper, discovery_operation("mysql")), (db_tools.table_wrapper, evidence_operation("mysql")),
                (ragflow_tools.assistant_wrapper, discovery_operation("ragflow")), (ragflow_tools.ask_wrapper, evidence_operation("ragflow"))]
    try:
        with contextlib.ExitStack() as stack:
            for wrapper, operation in wrappers:
                stack.enter_context(patch.object(wrapper, "operation", operation))
                stack.enter_context(patch.object(wrapper, "policy", RetryPolicy(timeout_s=3, max_attempts=wrapper.policy.max_attempts, delay_s=0)))
            events = stack.enter_context(patch("app.api.monitor.monitor._emit"))
            result = await agent.ainvoke({"messages": [{"role": "user", "content": case["query"]}]})
        messages = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.tool_call_id.startswith("task-")]
        if len(run.context_counts) != len(phases) + 1:
            raise AssertionError("Supervisor model calls must match the frozen phase plan")
        ingress = injected_message_volume(model.seen_messages)
        provider_observations = []
        for source, expert in experts.items():
            seen = set()
            for snapshot in expert.seen_messages:
                for message in snapshot:
                    if isinstance(message, ToolMessage) and message.id not in seen:
                        seen.add(message.id)
                        provider_observations.append({"source": source, "units": counter.count(message.content)})
        grouped = [Attempt(source, [row for row in run.results if row.source == source]) for source in case["sources"]]
        outcome = ResearchOutcome(grouped, case["minimum_sources"])
        delegations = [dict(call.args[2]) for call in events.call_args_list
                       if len(call.args) == 3 and call.args[0] == "assistant_progress" and call.args[2].get("status") == "started"]
        return {"model_call_input_tokens_proxy": run.context_counts, "mean_input_tokens_proxy": mean(run.context_counts),
            "cumulative_input_tokens_proxy": sum(run.context_counts), "final_input_tokens_proxy": run.context_counts[-1],
            "effective_context_units": ingress["total_units"], "injected_messages": ingress["messages"],
            "provider_observations": provider_observations, "delegations": delegations,
            "completed": outcome.completed, "failures": outcome.failures,
            "subtask_sizes": [counter.count(m.content) for m in messages],
            "required_fact_retained": all(any(text in m.content for m in messages) for text in case["expected"]["required_text"])}
    finally:
        run_context.reset(token)


def code_fingerprint():
    digest = hashlib.sha256()
    paths = list((ROOT / "app").rglob("*.py")) + list((ROOT / "skills").rglob("SKILL.md"))
    paths += [ROOT / "app/prompt/prompts.yml", ROOT / "eval/run_deep_research_benchmark.py", ROOT / "eval/baseline.py"]
    paths += [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    paths += list((ROOT / "frontend/src").rglob("*.ts")) + list((ROOT / "frontend/src").rglob("*.tsx")) + list((ROOT / "frontend/src").rglob("*.css"))
    paths += list((ROOT / "tests").rglob("*.py")) + [ROOT / ".gitignore"]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


async def run_suite(cases=None, context_limit=EFFECTIVE_CONTEXT_LIMIT):
    cases = cases or load_dataset()
    rows = []
    for case in cases:
        result = await run_case(case)
        rows.append({"id": case["id"], "category": case["category"], "passed": grade_case(case, result), **result})
    fault_cases = [c for c in cases if c["category"] == "single_source_fault_injection"]
    experiment_a = []
    for case in fault_cases:
        current = evaluate_baseline(case, {s: fixture_operation(case, s) for s in case["sources"]})
        unprotected = evaluate_baseline(case, {s: fixture_operation(case, s) for s in case["sources"]}, unprotected=True)
        wrapped = await replay_supervisor_context(case, True)
        experiment_a.append({"id": case["id"], "faults": case["faults"], "current_tool_baseline": current,
            "unprotected": unprotected, "wrapper_completed": wrapped["completed"], "failures": wrapped["failures"],
            "actual_delegations": wrapped["delegations"]})
    experiment_b = []
    for case in cases:
        if case["category"] != "normal_multi_source":
            continue
        before = await replay_supervisor_context(case, False)
        after = await replay_supervisor_context(case, True)
        experiment_b.append({"id": case["id"], "before": before, "after": after,
            "profile": case["context_workload"]["profile"],
            "context_over_budget_before": before["effective_context_units"] > context_limit,
            "context_over_budget_after": after["effective_context_units"] > context_limit})
    before_mean = mean(row["before"]["effective_context_units"] for row in experiment_b)
    after_mean = mean(row["after"]["effective_context_units"] for row in experiment_b)
    head = subprocess.run(["git", "-c", f"safe.directory={ROOT.as_posix()}", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return {"schema_version": 2, "baseline_commit": BASELINE_COMMIT, "checkout_head": head.stdout.strip() if head.returncode == 0 else "unavailable",
        "code_sha256": code_fingerprint(), "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "tokenizer": counter.name,
        "count_scope": "per-task cumulative injection of new non-system Supervisor messages; each message counted once; fixed system and tool definitions excluded",
        "measurement": {"primary": "sum(injected_messages[].units)", "unit": counter.name,
            "included": ["user query", "Supervisor task call arguments", "returned task evidence and status envelopes"],
            "excluded": ["fixed system prompts", "tool definitions", "replayed copies of existing message history", "final output not fed back as input"],
            "auxiliary": "full model inputs retained in experiment_b.before/after.model_call_input_tokens_proxy",
            "trajectory": "discovery: tavily/mysql/ragflow; verification: tavily/ragflow; three Supervisor calls",
            "fixture_scope": "synthetic stratified stress fixtures; compact/standard/long = 3/6/3",
            "budget_stages": {"provider_observation": 6000, "delegation_result": 1500},
            "overflow_rule": "effective_context_units > effective_context_limit"},
        "effective_context_limit": context_limit, "case_count": len(rows), "passed": sum(r["passed"] for r in rows),
        "cases": rows, "experiment_a": experiment_a, "experiment_b": experiment_b,
        "summary": {"current_tool_baseline_success": sum(r["current_tool_baseline"]["completed"] for r in experiment_a),
                    "unprotected_success": sum(r["unprotected"]["completed"] for r in experiment_a),
                    "wrapper_success": sum(r["wrapper_completed"] for r in experiment_a),
                    "mean_effective_context_before": before_mean, "mean_effective_context_after": after_mean,
                    "mean_full_model_input_before": mean(r["before"]["mean_input_tokens_proxy"] for r in experiment_b),
                    "mean_full_model_input_after": mean(r["after"]["mean_input_tokens_proxy"] for r in experiment_b),
                    "reduction_percent": (before_mean - after_mean) / before_mean * 100,
                    "context_over_budget_before": sum(r["context_over_budget_before"] for r in experiment_b),
                    "context_over_budget_after": sum(r["context_over_budget_after"] for r in experiment_b)}}


def render_report(results):
    s = results["summary"]
    lines = ["# Deep Research Agent Harness 基准评测报告", "",
        "本地全量验证结果。评测通过Mock故障注入、脚本模型与真实DeepAgents图验证工具契约、跨源恢复和证据预算。", "",
        f"- 样本：{results['case_count']}；行为断言通过：{results['passed']}。",
        f"- 对照提交：`{results['baseline_commit']}`；当前HEAD：`{results['checkout_head']}`。",
        f"- 代码SHA-256：`{results['code_sha256']}`。",
        f"- 数据集SHA-256：`{results['dataset_sha256']}`。",
        f"- 计数器：`{results['tokenizer']}`。", "",
        "## 实验 A：容错与跨源恢复", "",
        "| 执行策略 | 完成数 | 完成率 |", "|---|---:|---:|",
        f"| 基础异常捕获，无跨源恢复策略 | {s['current_tool_baseline_success']}/7 | {s['current_tool_baseline_success']/7:.1%} |",
        f"| ToolWrapper 错误分类 + DelegationHarnessMiddleware 跨源恢复 | {s['wrapper_success']}/7 | {s['wrapper_success']/7:.1%} |", "",
        "评测要求至少两个不同可用来源类别提供证据。基础组执行已有工具异常捕获逻辑；增强组运行真实专家工具、ProviderHarnessMiddleware与DelegationHarnessMiddleware，逐例保留实际委派轨迹。", "",
        "| 样本 | 注入故障 | 基础组 | 增强组 | 实际专家委派次数 |", "|---|---|---|---|---:|"]
    for row in results["experiment_a"]:
        before = "完成" if row["current_tool_baseline"]["completed"] else "中止"
        after = "完成" if row["wrapper_completed"] else "安全中止"
        lines.append(f"| {row['id']} | {json.dumps(row['faults'], ensure_ascii=False)} | {before} | {after} | {len(row['actual_delegations'])} |")
    lines += ["", "F07同时失去Tavily与RAGFlow，仅剩MySQL，按双来源证据要求安全中止；错误类型、来源与委派轨迹记录在results.json。",
        f"辅助诊断：显式无捕获对照完成{s['unprotected_success']}/7，完整数据保存在experiment_a[].unprotected。", "",
        "## 实验 B：两级预算与证据去重", "",
        "### 计量口径", "",
        "主指标为**单次调研任务累积注入主控的证据与消息上下文体积**。每条首次进入主控输入的user/assistant/tool消息计一次；不同工具返回中重复出现的证据仍计入实际注入体积。固定系统提示词和工具定义不纳入主指标，同一历史消息在后续模型调用中重放时不重复计量。", "",
        "计算式：`effective_context_units = sum(injected_messages[].units)`；`units = counter.count(canonical_json(role, content, tool_calls))`。消息ID只用于识别历史重放，不参与计数；保留每条消息的计数和SHA-256以便审计。", "",
        "样本为分层合成压力样本：紧凑3例、标准长页6例、扩展长页3例，包含追踪URL镜像、相似正文和重复核验检索。工作流固定为首轮三源取证、第二轮网页与文档核验，共5个子任务、3次主控模型调用。样本分层、检索计划和阈值存于数据集，结果由同一计量函数直接计算。", "",
        "关闭组同时关闭工具观察与子任务返回裁剪；启用组使用真实ProviderHarnessMiddleware的6000输入保护、DelegationHarnessMiddleware的1500子任务预算和URL/内容指纹/SimHash去重。工具Schema、取证事实、问题与轨迹保持一致。", "",
        "| 指标 | 关闭裁剪 | 两级预算 + 去重 |", "|---|---:|---:|",
        f"| 平均每任务有效上下文体积（计数单位） | {s['mean_effective_context_before']:,.1f} | {s['mean_effective_context_after']:,.1f} |",
        f"| 上下文超阈次数 / 12 | {s['context_over_budget_before']} | {s['context_over_budget_after']} |",
        f"| 平均体积缩减 | — | {s['reduction_percent']:.2f}% |", "",
        f"阈值规则：`effective_context_units > {results['effective_context_limit']}`。该值是本压力集的动态上下文预算，超阈计数用于衡量预算压力；真实生成延迟和网络超时采用独立在线测量。", "",
        "| 样本 | 分层 | 关闭裁剪 | 两级预算 + 去重 | 超阈（前→后） | 关键事实保留 |", "|---|---|---:|---:|---|---|"]
    for row in results["experiment_b"]:
        lines.append(f"| {row['id']} | {row['profile']} | {row['before']['effective_context_units']} | {row['after']['effective_context_units']} | {row['context_over_budget_before']}→{row['context_over_budget_after']} | {row['after']['required_fact_retained']} |")
    lines += ["", "### 辅助全量输入统计", "",
        f"包含框架系统提示词、工具定义和重复历史的每次模型调用平均代理计数为{s['mean_full_model_input_before']:,.1f}→{s['mean_full_model_input_after']:,.1f}。逐轮、累计和最终调用值保存在experiment_b[].before/after，便于同时评估固定开销与动态证据。", "",
        "计数单位由tokenizer字段明确标识：有本地cl100k词表时使用该编码代理计数，缺失时使用UTF-8字节预算；实际API计费Token以目标模型usage为准。以上结果适用于公开的合成压力集及固定工作流。", "",
        "## 24例行为断言", "", "| 样本 | 分类 | 实际行为 | 断言通过 |", "|---|---|---|---|"]
    for row in results["cases"]:
        lines.append(f"| {row['id']} | {row['category']} | {row['behavior']} | {row['passed']} |")
    lines += ["", "## 验证边界", "",
        "- 执行闭环：Mock与脚本模型固定输入轨迹，真实运行工具封装、专家图、两级预算和跨源委派。",
        "- 事实质量：校验关键事实保留、冲突证据及未解决标记、SQL拒绝行为和私有事实来源边界。",
        "- 指标解释：任务完成率、行为断言通过率和上下文预算超阈分别统计。",
        "- 在线验证：模型回答质量、API计费、端到端延迟及真实超时率作为独立评测项。",
        "- 回归入口：python -m unittest覆盖工具适配、原生Skill、取消、并发隔离和HTTP/WebSocket。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--effective-context-limit", "--simulated-context-limit", dest="effective_context_limit", type=int, default=EFFECTIVE_CONTEXT_LIMIT)
    args = parser.parse_args()
    if args.effective_context_limit <= 0:
        parser.error("context limit must be positive")
    with offline_network(), contextlib.redirect_stdout(io.StringIO()):
        results = asyncio.run(run_suite(context_limit=args.effective_context_limit))
    report = render_report(results)
    (ROOT / "eval/results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ROOT / "eval/EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    s = results["summary"]
    print("Group                         Completed      Rate")
    for label, key in (("Basic catch, no recovery", "current_tool_baseline_success"), ("ToolWrapper + cross-source", "wrapper_success")):
        print(f"{label:29} {s[key]}/7            {s[key]/7:.1%}")
    print(f"Per-task dynamic context: {s['mean_effective_context_before']:,.1f} -> {s['mean_effective_context_after']:,.1f}; reduction {s['reduction_percent']:.2f}%")
    print(f"Counter: {results['tokenizer']}; context overflows {s['context_over_budget_before']}/12 -> {s['context_over_budget_after']}/12")
    print(f"Behavior checks: {results['passed']}/24; report: eval/EVALUATION_REPORT.md")
    return 0 if results["passed"] == 24 else 1


if __name__ == "__main__":
    raise SystemExit(main())
