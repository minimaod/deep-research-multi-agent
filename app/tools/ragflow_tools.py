"""RAGFlow adapters with references, bounded streaming and finally cleanup."""
import hashlib
import json
import time
import uuid
from langchain_core.tools import tool
from app.harness.runtime import check_cancelled
from app.harness.schemas import AnswerData, AskInput, AssistantData, AssistantInfo, EmptyInput, ErrorCode, EvidenceItem
from app.harness.tool_wrapper import RetryPolicy, ToolFailure, ToolWrapper, bind_async
from app.ragflow.rag_config import get_ragflow_client


def assistant_operation(params, timeout):
    client = get_ragflow_client(timeout)
    try:
        infos = [AssistantInfo(name=c.name, description=c.description or "", datasets=getattr(c, "kb_names", []) or []) for c in client.list_chats()]
        if not infos:
            raise ToolFailure(ErrorCode.EMPTY, "no_ragflow_assistants")
        return AssistantData(assistants=infos), [], []
    finally:
        client.close()


def ask_operation(params, timeout):
    client = get_ragflow_client(timeout)
    deadline = time.monotonic() + timeout
    session = chat = response = None
    warnings = []
    try:
        chats = client.list_chats(name=params.chat_name)
        if not chats:
            raise ToolFailure(ErrorCode.EMPTY, "assistant_not_found")
        chat = chats[0]
        session = chat.create_session(name="harness-" + uuid.uuid4().hex)
        response = client.post(f"/chats/{chat.id}/completions", {
            "messages": [{"role": "user", "content": params.question}], "stream": True, "session_id": session.id}, stream=True)
        result, refs = "", {}
        for line in response.iter_lines(decode_unicode=True):
            check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError("ragflow stream deadline")
            if not line or line.startswith(":"):
                continue
            line = line.removeprefix("data:").strip()
            if line == "[DONE]":
                break
            payload = json.loads(line)
            if payload.get("code", 0) != 0:
                raise ToolFailure(ErrorCode.UNAVAILABLE, "ragflow_stream_error")
            data = payload.get("data")
            if not isinstance(data, dict):
                continue
            answer = data.get("answer")
            if answer:
                if not isinstance(answer, str):
                    raise ToolFailure(ErrorCode.UNAVAILABLE, "invalid_ragflow_answer")
                if answer.startswith(result):
                    result = answer
                elif not result.startswith(answer):
                    result += answer
            reference = data.get("reference") or {}
            if isinstance(reference, dict):
                for chunk in reference.get("chunks", []):
                    text = chunk.get("content") or chunk.get("content_with_weight")
                    if text:
                        key = str(chunk.get("id") or hashlib.sha256(text.encode()).hexdigest()[:16])
                        locator = str(chunk.get("url") or f"ragflow:document:{chunk.get('document_id', '')}:chunk:{key}")
                        refs[key] = EvidenceItem(id="rag-" + key[:100], source="ragflow", text=text,
                            title=str(chunk.get("document_name") or "")[:300], locator=locator[:2048])
        if not result.strip():
            raise ToolFailure(ErrorCode.EMPTY, "empty_ragflow_answer")
        evidence = list(refs.values())
        if not evidence:
            warnings.append("references_missing: answer is unverified, do not cite as a document quote")
            evidence = [EvidenceItem(id="rag-answer-" + uuid.uuid4().hex[:16], source="ragflow", text=result,
                locator=f"ragflow:unverified-answer:{params.chat_name}")]
        data = AnswerData(answer=result, references=[e.locator for e in refs.values()], references_missing=not bool(refs))
        return data, evidence, warnings
    finally:
        if response is not None:
            response.close()
        if session is not None and chat is not None:
            try:
                if hasattr(client, "deadline"):
                    client.deadline = time.monotonic() + 2
                chat.delete_sessions(ids=[session.id])
            except Exception:
                warnings.append("temporary_session_cleanup_failed")
                from app.api.monitor import monitor
                monitor._emit("tool_cleanup_warning", "RAGFlow 临时会话清理失败", {"tool_name": "create_ask_delete"})
        client.close()


assistant_wrapper = ToolWrapper("ragflow", "get_assistant_list", EmptyInput, assistant_operation)
# Session creation is not idempotent: never replay it automatically.
ask_wrapper = ToolWrapper("ragflow", "create_ask_delete", AskInput, ask_operation, policy=RetryPolicy(max_attempts=1))


@tool(args_schema=EmptyInput)
def get_assistant_list() -> dict:
    """发现RAGFlow助手及绑定知识库；无助手返回EmptyResult降级。"""
    return assistant_wrapper.execute({}).model_dump(mode="json")


@tool(args_schema=AskInput)
def create_ask_delete(chat_name: str, question: str) -> dict:
    """向已发现助手问答，保留引用；任何出口均尝试清理临时会话。"""
    return ask_wrapper.execute({"chat_name": chat_name, "question": question}).model_dump(mode="json")


bind_async(get_assistant_list, assistant_wrapper)
bind_async(create_ask_delete, ask_wrapper)
