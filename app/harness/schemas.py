"""Provider-independent contracts. Provider wire payloads are adapted at the edge."""
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ErrorCode(StrEnum):
    TIMEOUT = "TimeoutError"
    UNAVAILABLE = "ServiceUnavailable"
    EMPTY = "EmptyResult"
    INVALID = "InvalidParam"


Source = Literal["tavily", "mysql", "ragflow"]
ShortText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=12000)]


class SearchInput(Contract):
    query: ShortText
    topic: Literal["news", "finance", "general"] = "general"
    max_results: Annotated[int, Field(strict=True, ge=1, le=20)] = 5
    include_raw_content: Annotated[bool, Field(strict=True)] = False


class EmptyInput(Contract):
    pass


class TableInput(Contract):
    table_name: Annotated[str, Field(strict=True, pattern=r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")]


class SQLInput(Contract):
    query: ShortText


class AskInput(Contract):
    chat_name: Annotated[str, Field(strict=True, min_length=1, max_length=200)]
    question: ShortText


class EvidenceItem(Contract):
    id: str = Field(min_length=1, max_length=160)
    source: Source
    text: str = Field(min_length=1)
    locator: str = Field(default="", max_length=2048)
    title: str = Field(default="", max_length=300)
    observed_at: str = Field(default="", max_length=80)
    score: float = Field(default=0.0, ge=0, le=1)
    # A row stays atomic; prose can be split at sentence boundaries.
    kind: Literal["prose", "row"] = "prose"
    aliases: list[str] = Field(default_factory=list)


class SearchHit(Contract):
    title: str
    url: str
    content: str
    score: float = Field(ge=0, le=1)


class SearchData(Contract):
    kind: Literal["search"] = "search"
    query: str
    results: list[SearchHit]


class SQLData(Contract):
    kind: Literal["sql"] = "sql"
    columns: list[str]
    # Connector values are normalized to JSON strings at the edge, including NULL.
    rows: list[list[str | None]]
    truncated: bool = False

    @model_validator(mode="after")
    def rectangular(self):
        if any(len(row) != len(self.columns) for row in self.rows):
            raise ValueError("SQL row width must match columns")
        return self


class AssistantInfo(Contract):
    name: str
    description: str
    datasets: list[str]


class AssistantData(Contract):
    kind: Literal["assistants"] = "assistants"
    assistants: list[AssistantInfo]


class AnswerData(Contract):
    kind: Literal["answer"] = "answer"
    answer: str
    references: list[str]
    references_missing: bool


ToolData = Annotated[SearchData | SQLData | AssistantData | AnswerData, Field(discriminator="kind")]


class ToolError(Contract):
    code: ErrorCode
    message: str
    retryable: bool = False
    reason: str = ""


class ToolExecutionResult(Contract):
    schema_version: Literal["1"] = "1"
    source: Source
    tool_name: str
    call_id: str
    status: Literal["ok", "degraded"]
    data: ToolData | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    error: ToolError | None = None
    attempts: int = Field(default=1, ge=0)
    cost_ms: float = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_status(self):
        if (self.status == "degraded") != (self.error is not None):
            raise ValueError("degraded requires error; ok forbids error")
        if self.status == "ok" and self.data is None:
            raise ValueError("ok requires typed data")
        if self.status == "degraded" and (self.data is not None or self.evidence):
            raise ValueError("failed calls cannot provide successful evidence")
        if self.data and self.data.kind not in {"tavily": {"search"}, "mysql": {"sql"}, "ragflow": {"answer", "assistants"}}[self.source]:
            raise ValueError("provider and data kind must agree")
        if any(item.source != self.source for item in self.evidence):
            raise ValueError("evidence must belong to the producing source")
        return self
