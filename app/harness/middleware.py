"""Intercept actual ToolMessage/Command results before the next model invocation."""
import asyncio
import dataclasses
import json
import uuid
from pathlib import Path

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from pydantic import ValidationError

from app.harness.context_trimmer import render_evidence, trim_context
from app.harness.research_policy import AGENT_SOURCES, SOURCE_AGENTS, Attempt, ResearchRecoveryPolicy
from app.harness.runtime import SubtaskState, check_cancelled, subtask_context
from app.harness.schemas import ErrorCode, ToolError, ToolExecutionResult
from app.harness.token_counter import count_model_context, counter

PROVIDER_TOOLS = {"internet_search": "tavily", "list_sql_tables": "mysql", "get_table_data": "mysql", "execute_sql_query": "mysql", "get_assistant_list": "ragflow", "create_ask_delete": "ragflow"}


class ContextUsageMiddleware(AgentMiddleware):
    """Measure the Supervisor input after other middleware has built its prompt."""
    def _record(self, request):
        from app.api.monitor import monitor
        from app.harness.runtime import run_context
        tools = [{"name": t.name, "description": t.description, "parameters": t.args}
                 if hasattr(t, "name") else t for t in request.tools]
        count = count_model_context(request.messages, request.system_message.content if request.system_message else "", tools)
        state = run_context.get()
        if state:
            state.context_counts.append(count)
        monitor._emit("context_usage", "主控输入上下文计量", {"input_tokens_proxy": count, "tokenizer": counter.name})

    def wrap_model_call(self, request, handler):
        self._record(request)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        self._record(request)
        return await handler(request)


def replace_message(result, content):
    """Preserve command routing, files, IDs, artifacts and all unrelated updates."""
    if isinstance(result, Command):
        update = dict(result.update or {})
        messages = list(update.get("messages", []))
        if not messages:
            raise ValueError("task Command did not contain a message")
        messages[-1] = messages[-1].model_copy(update={"content": content})
        update["messages"] = messages
        return dataclasses.replace(result, update=update)
    return result.model_copy(update={"content": content})


def combine_task_updates(attempts):
    primary = attempts[0].value
    if not isinstance(primary, Command):
        return primary
    update = dict(primary.update or {})
    for attempt in attempts[1:]:
        if not isinstance(attempt.value, Command):
            continue
        for key, value in (attempt.value.update or {}).items():
            if key == "messages":
                continue
            if isinstance(update.get(key), dict) and isinstance(value, dict):
                update[key] = {**update[key], **value}
            else:
                update[key] = value
    return dataclasses.replace(primary, update=update)


def tool_failure_message(request, code, reason):
    source = PROVIDER_TOOLS[request.tool_call["name"]]
    result = ToolExecutionResult(source=source, tool_name=request.tool_call["name"], call_id=request.tool_call["id"],
        status="degraded", attempts=0, error=ToolError(code=code, message=reason, reason=reason))
    from app.harness.runtime import record_result
    from app.api.monitor import monitor
    record_result(result)
    monitor._emit("tool_degraded", "工具参数校验失败", {"call_id": result.call_id, "error": result.error.model_dump(mode="json")})
    return ToolMessage(result.model_dump_json(), tool_call_id=request.tool_call["id"])


class ProviderHarnessMiddleware(AgentMiddleware):
    """Guard raw tool observations sent to child models, without losing raw ledger."""
    def __init__(self, max_tokens=6000, trim_enabled=True):
        self.max_tokens = max_tokens
        self.trim_enabled = trim_enabled

    def _observe(self, result, request):
        if request.tool_call["name"] not in PROVIDER_TOOLS:
            return result
        if isinstance(result, ToolMessage) and result.status == "error":
            return tool_failure_message(request, ErrorCode.INVALID, "tool_input_rejected")
        if not self.trim_enabled:
            return result
        try:
            parsed = ToolExecutionResult.model_validate_json(result.content)
        except (ValueError, TypeError, ValidationError):
            return result
        # Discovery data is needed for routing; cap names and descriptions, not facts.
        metadata = {"schema_version": "1", "source": parsed.source, "tool_name": parsed.tool_name,
                    "status": parsed.status, "call_id": parsed.call_id,
                    "error": parsed.error.model_dump(mode="json") if parsed.error else None,
                    "warnings": parsed.warnings}
        if parsed.data and parsed.data.kind == "assistants":
            metadata["assistants"] = []
            for assistant in parsed.data.assistants[:8]:
                entry = {"name": assistant.name, "description": assistant.description[:120]}
                trial = {**metadata, "assistants": metadata["assistants"] + [entry]}
                if counter.count(json.dumps(trial, ensure_ascii=False)) > 4500:
                    break
                metadata["assistants"].append(entry)
        evidence = parsed.evidence
        clipped = trim_context(evidence, json.dumps(request.tool_call["args"], ensure_ascii=False), self.max_tokens, metadata)
        if evidence and not clipped.evidence:
            metadata.update(status="degraded", error={"code": "EmptyResult", "reason": "evidence_exceeds_budget"})
            clipped = trim_context([], "", self.max_tokens, metadata)
        return replace_message(result, clipped.content)

    def wrap_tool_call(self, request, handler):
        check_cancelled()
        try:
            result = handler(request)
        except ValidationError:
            if request.tool_call["name"] not in PROVIDER_TOOLS:
                raise
            return tool_failure_message(request, ErrorCode.INVALID, "input_validation_failed")
        return self._observe(result, request)

    async def awrap_tool_call(self, request, handler):
        check_cancelled()
        try:
            result = await handler(request)
        except ValidationError:
            if request.tool_call["name"] not in PROVIDER_TOOLS:
                raise
            return tool_failure_message(request, ErrorCode.INVALID, "input_validation_failed")
        return self._observe(result, request)


class DelegationHarnessMiddleware(AgentMiddleware):
    """Budget every registered research subtask and deterministically try compatible fallback."""
    def __init__(self, max_tokens=1500, trim_enabled=True):
        self.max_tokens = max_tokens
        self.trim_enabled = trim_enabled
        self.policy = ResearchRecoveryPolicy()

    async def awrap_tool_call(self, request, handler):
        denied = guard_session_files(request)
        if denied is not None:
            return denied
        if request.tool_call["name"] != "task" or request.tool_call["args"].get("subagent_type") not in AGENT_SOURCES:
            check_cancelled()
            return await handler(request)
        from app.api.monitor import monitor
        from app.api.context import get_session_context
        query = request.tool_call["args"]["description"]
        primary = AGENT_SOURCES[request.tool_call["args"]["subagent_type"]]

        async def invoke(source):
            check_cancelled()
            state = SubtaskState(uuid.uuid4().hex, source)
            token = subtask_context.set(state)
            try:
                monitor._emit("assistant_progress", f"委派 {SOURCE_AGENTS[source]}", {"assistant_name": SOURCE_AGENTS[source], "status": "started", "call_id": request.tool_call["id"]})
                args = {**request.tool_call["args"], "subagent_type": SOURCE_AGENTS[source], "description": query}
                value = await handler(request.override(tool_call={**request.tool_call, "args": args}))
                check_cancelled()
                monitor._emit("assistant_progress", f"{SOURCE_AGENTS[source]} 返回", {"assistant_name": SOURCE_AGENTS[source], "status": "finished", "call_id": request.tool_call["id"]})
                return Attempt(source, list(state.results), value)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                from app.harness.tool_wrapper import classify_error
                from app.harness.runtime import record_result
                failure = ToolExecutionResult(source=source, tool_name="subagent_execution", call_id=request.tool_call["id"],
                    status="degraded", error=classify_error(exc))
                record_result(failure)
                monitor._emit("assistant_progress", f"{SOURCE_AGENTS[source]} 异常，检查适用替代来源", {"assistant_name": SOURCE_AGENTS[source], "status": "degraded", "error": failure.error.model_dump(mode="json")})
                return Attempt(source, list(state.results), ToolMessage(failure.model_dump_json(), tool_call_id=request.tool_call["id"]))
            finally:
                subtask_context.reset(token)

        outcome = await self.policy.collect(query, [primary], invoke,
            allowed_sources=self.policy.compatible_sources(query, primary))
        # Archive raw evidence outside the model context, scoped to the current session.
        directory = get_session_context()
        if directory:
            archive = Path(directory) / ".harness_evidence"
            archive.mkdir(exist_ok=True)
            (archive / f"{uuid.uuid4().hex}.json").write_text(json.dumps([e.model_dump(mode="json") for e in outcome.evidence], ensure_ascii=False), encoding="utf-8")
        if self.trim_enabled:
            clipped = trim_context(outcome.evidence, query, self.max_tokens, outcome.metadata())
            if outcome.evidence and not clipped.evidence:
                metadata = {**outcome.metadata(), "status": "degraded", "outcome": "aborted", "missing_evidence": ["evidence_exceeds_budget"]}
                clipped = trim_context([], query, self.max_tokens, metadata)
            content = clipped.content
            monitor._emit("context_trimmed", "子任务证据已去重并按预算返回", {"call_id": request.tool_call["id"], **clipped.stats()})
        else:
            # Explicit ablation switch; production leaves this enabled.
            content = render_evidence(outcome.evidence, outcome.metadata())
        # Original call ID and Command side effects remain intact even after fallback.
        return replace_message(combine_task_updates(outcome.attempts), content)

    def wrap_tool_call(self, request, handler):
        # DeepAgents app uses astream. Sync support uses the same policy and preserves
        # the ContextVars copied by LangChain's sync executor.
        async def adapted(req):
            return handler(req)
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.awrap_tool_call(request, adapted))
        raise RuntimeError("Use ainvoke/astream inside an active asyncio loop")


def guard_session_files(request):
    """Enforce the existing session-directory promise on the actual file tools."""
    from app.api.context import get_session_context
    from app.utils.path_utils import resolve_path
    name, args = request.tool_call["name"], request.tool_call["args"]
    directory = get_session_context()
    if not directory or name not in {"read_file_content", "generate_markdown", "convert_md_to_pdf"}:
        return None
    paths = []
    if name == "generate_markdown":
        paths = [str(Path(args.get("path") or ".") / args.get("filename", ""))]
    elif name == "read_file_content":
        paths = [args.get("filename", "")]
    else:
        paths = [args.get("md_filename", "")] + ([args["pdf_filename"]] if args.get("pdf_filename") else [])
    if any(not Path(resolve_path(path, directory)).resolve().is_relative_to(Path(directory).resolve()) for path in paths):
        return ToolMessage('{"status":"degraded","error":{"code":"InvalidParam","reason":"outside_session_directory"}}', tool_call_id=request.tool_call["id"])
    return None
