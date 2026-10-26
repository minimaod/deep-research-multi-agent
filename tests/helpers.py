from app.harness.schemas import EvidenceItem, SearchData, SearchHit, SQLData, AnswerData, ToolExecutionResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from pydantic import Field


class ScriptedModel(FakeMessagesListChatModel):
    """A scripted tool-calling model: exercises graph wiring, not LLM intelligence."""
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def evidence(source="tavily", text="Revenue increased to 20 million in 2025.", id="e1", **kwargs):
    return EvidenceItem(id=id, source=source, text=text, locator=f"https://example.invalid/{id}", **kwargs)


def payload(source="tavily", text="Grounded result."):
    item = evidence(source, text)
    if source == "tavily":
        data = SearchData(query="research", results=[SearchHit(title="Test", url=item.locator, content=text, score=.9)])
    elif source == "mysql":
        data = SQLData(columns=["finding"], rows=[[text]])
    else:
        data = AnswerData(answer=text, references=[item.locator], references_missing=False)
    return data, [item], []


def success(source="tavily", text="Grounded result.", tool="research"):
    data, items, _ = payload(source, text)
    return ToolExecutionResult(source=source, tool_name=tool, call_id="call-test", status="ok", data=data, evidence=items)
