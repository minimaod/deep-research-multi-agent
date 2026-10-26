"""Run/subtask state carried through asyncio tasks and executor context copies."""
import asyncio
import threading
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field


@dataclass
class RunState:
    thread_id: str
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    cancelled: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    workers: int = 0
    results: list = field(default_factory=list)
    context_counts: list = field(default_factory=list)

    def enter_worker(self):
        with self.lock:
            if self.cancelled.is_set():
                raise asyncio.CancelledError()
            self.workers += 1

    def leave_worker(self):
        with self.lock:
            self.workers -= 1

    async def wait_idle(self):
        # Native provider requests have bounded connect/read timeouts. Keep the
        # session occupied until they finish rather than pretending to kill threads.
        while True:
            with self.lock:
                if self.workers == 0:
                    return
            await asyncio.sleep(.01)


@dataclass
class SubtaskState:
    id: str
    source: str
    results: list = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)


run_context: ContextVar[RunState | None] = ContextVar("harness_run", default=None)
subtask_context: ContextVar[SubtaskState | None] = ContextVar("harness_subtask", default=None)
active_runs: dict[str, RunState] = {}


def check_cancelled():
    run = run_context.get()
    if run and run.cancelled.is_set():
        raise asyncio.CancelledError()


def record_result(result):
    check_cancelled()
    run = run_context.get()
    if run:
        with run.lock:
            run.results.append(result)
    subtask = subtask_context.get()
    if subtask:
        with subtask.lock:
            subtask.results.append(result)


def event_context():
    run, subtask = run_context.get(), subtask_context.get()
    return {**({"run_id": run.run_id, "thread_id": run.thread_id} if run else {}),
            **({"subtask_id": subtask.id} if subtask else {})}


def accept_event(event):
    run = run_context.get()
    if not run:
        return True
    current = active_runs.get(run.thread_id)
    return (current is None or current is run) and (not run.cancelled.is_set() or event == "task_cancelled")
