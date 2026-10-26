"""One error taxonomy and bounded retry policy for all provider adapters."""
import asyncio
import threading
import time
import uuid
from dataclasses import dataclass

from pydantic import ValidationError

from app.harness.runtime import check_cancelled, record_result, run_context
from app.harness.schemas import ErrorCode, ToolError, ToolExecutionResult


class ToolFailure(Exception):
    def __init__(self, code, reason, retryable=False):
        self.code, self.reason, self.retryable = code, reason, retryable
        super().__init__(reason)


@dataclass(frozen=True)
class RetryPolicy:
    timeout_s: float = 20.0
    max_attempts: int = 2
    delay_s: float = .1

    def __post_init__(self):
        if self.timeout_s <= 0 or self.max_attempts not in (1, 2) or self.delay_s < 0:
            raise ValueError("positive deadline and at most two attempts required")


def classify_error(exc):
    if isinstance(exc, ToolFailure):
        return ToolError(code=exc.code, message=exc.reason, reason=exc.reason, retryable=exc.retryable)
    name = type(exc).__name__.lower()
    errno = getattr(exc, "errno", None)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(exc, TimeoutError) or "timeout" in name or errno in {3024}:
        code, retryable, reason = ErrorCode.TIMEOUT, True, "provider_timeout"
    elif errno in {1054, 1064, 1146}:
        code, retryable, reason = ErrorCode.INVALID, False, "invalid_query"
    elif isinstance(exc, ValidationError):
        code, retryable, reason = ErrorCode.UNAVAILABLE, False, "invalid_provider_output"
    elif status in {400, 422}:
        code, retryable, reason = ErrorCode.INVALID, False, "provider_rejected_parameters"
    else:
        code = ErrorCode.UNAVAILABLE
        retryable = isinstance(exc, (ConnectionError, OSError)) or "connection" in name or status in {429, 500, 502, 503, 504} or errno in {2003, 2006, 2013}
        reason = "authentication_or_configuration" if status in {401, 403} else "provider_unavailable"
    # Never forward raw provider exception strings (may contain keys, SQL or URLs).
    return ToolError(code=code, message=reason, reason=reason, retryable=retryable)


class ToolWrapper:
    def __init__(self, source, tool_name, input_schema, operation, policy=None, emit=None):
        self.source, self.tool_name = source, tool_name
        self.input_schema, self.operation = input_schema, operation
        self.policy = policy or RetryPolicy()
        self.emit = emit

    def _emit(self, event, message, data):
        if self.emit:
            self.emit(event, message, data)
        else:
            from app.api.monitor import monitor
            monitor._emit(event, message, data)

    def _finish(self, result, abandoned=None):
        check_cancelled()
        if abandoned and abandoned.is_set():
            return result
        record_result(result)
        data = {"tool_name": self.tool_name, "source": self.source,
                "call_id": result.call_id, "cost_ms": result.cost_ms,
                "attempts": result.attempts, "status": result.status}
        if result.error:
            data["error"] = result.error.model_dump(mode="json")
            self._emit("tool_degraded", f"{self.tool_name} 已降级：{result.error.code}", data)
        self._emit("tool_end", f"工具结束：{self.tool_name}", data)
        return result

    def execute(self, args, *, abandoned=None, call_id=None):
        check_cancelled()
        run = run_context.get()
        if run:
            run.enter_worker()
        try:
            return self._execute(args, abandoned, call_id)
        finally:
            if run:
                run.leave_worker()

    def _execute(self, args, abandoned, call_id=None):
        start = time.monotonic()
        call_id = call_id or uuid.uuid4().hex
        self._emit("tool_start", f"开始执行工具：{self.tool_name}", {"tool_name": self.tool_name, "source": self.source, "call_id": call_id})
        try:
            params = self.input_schema.model_validate(args)
        except ValidationError:
            return self._finish(ToolExecutionResult(source=self.source, tool_name=self.tool_name, call_id=call_id, status="degraded", attempts=0,
                error=ToolError(code=ErrorCode.INVALID, message="input_validation_failed", reason="input_validation_failed")), abandoned)
        error = None
        for attempt in range(1, self.policy.max_attempts + 1):
            check_cancelled()
            remaining = self.policy.timeout_s - (time.monotonic() - start)
            if remaining <= 0 or (abandoned and abandoned.is_set()):
                error = ToolError(code=ErrorCode.TIMEOUT, message="total_deadline_exceeded", reason="total_deadline_exceeded")
                break
            try:
                data, evidence, warnings = self.operation(params, remaining)
                check_cancelled()
                if time.monotonic() - start > self.policy.timeout_s:
                    raise ToolFailure(ErrorCode.TIMEOUT, "total_deadline_exceeded")
                result = ToolExecutionResult(source=self.source, tool_name=self.tool_name, call_id=call_id,
                    status="ok", data=data, evidence=evidence, warnings=warnings, attempts=attempt,
                    cost_ms=(time.monotonic() - start) * 1000)
                return self._finish(result, abandoned)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                error = classify_error(exc)
                if not error.retryable or attempt == self.policy.max_attempts:
                    break
                remaining = self.policy.timeout_s - (time.monotonic() - start)
                if remaining <= self.policy.delay_s:
                    break
                run = run_context.get()
                if run:
                    if run.cancelled.wait(self.policy.delay_s):
                        raise asyncio.CancelledError()
                else:
                    time.sleep(self.policy.delay_s)
        return self._finish(ToolExecutionResult(source=self.source, tool_name=self.tool_name, call_id=call_id,
            status="degraded", error=error, attempts=attempt, cost_ms=(time.monotonic() - start) * 1000), abandoned)

    async def aexecute(self, args):
        """Async deadline cannot kill a thread; RunState tracks physical cleanup."""
        abandoned = threading.Event()
        call_id = uuid.uuid4().hex
        start = time.monotonic()
        future = asyncio.create_task(asyncio.to_thread(self.execute, args, abandoned=abandoned, call_id=call_id))
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.policy.timeout_s)
        except asyncio.CancelledError:
            abandoned.set()
            future.add_done_callback(_consume_future)
            raise
        except TimeoutError:
            abandoned.set()
            future.add_done_callback(_consume_future)
            return self._finish(ToolExecutionResult(source=self.source, tool_name=self.tool_name,
                call_id=call_id, status="degraded", cost_ms=(time.monotonic() - start) * 1000,
                error=ToolError(code=ErrorCode.TIMEOUT, message="async_deadline_exceeded", reason="async_deadline_exceeded")))


def _consume_future(future):
    if not future.cancelled():
        future.exception()


def bind_async(tool, wrapper):
    """Preserve .invoke while making .ainvoke use an async deadline."""
    async def execute(**kwargs):
        return (await wrapper.aexecute(kwargs)).model_dump(mode="json")
    tool.coroutine = execute
    return tool
