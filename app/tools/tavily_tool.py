"""Lazy Tavily adapter with typed results, native timeout and bounded recovery."""
import hashlib
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient
from app.harness.schemas import ErrorCode, EvidenceItem, SearchData, SearchHit, SearchInput
from app.harness.tool_wrapper import ToolFailure, ToolWrapper, bind_async
load_dotenv()


def get_tavily_client():
    if not os.getenv("TAVILY_API_KEY"):
        raise ToolFailure(ErrorCode.UNAVAILABLE, "missing_tavily_configuration")
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def search_operation(params, timeout):
    client = get_tavily_client()
    try:
        payload = client.search(**params.model_dump(), timeout=timeout)
    finally:
        client.close()
    hits, evidence = [], []
    for item in payload["results"][:params.max_results]:
        text = item.get("raw_content") or item.get("content") or ""
        if not text.strip():
            continue
        hit = SearchHit(title=item.get("title", ""), url=item["url"], content=text, score=float(item.get("score", 0)))
        hits.append(hit)
        evidence.append(EvidenceItem(id="web-" + hashlib.sha256((hit.url + text).encode()).hexdigest()[:16],
            source="tavily", text=text, locator=hit.url, title=hit.title[:300], score=hit.score,
            observed_at=str(item.get("published_date") or "")[:80]))
    if not hits:
        raise ToolFailure(ErrorCode.EMPTY, "no_search_results")
    return SearchData(query=params.query, results=hits), evidence, []


search_wrapper = ToolWrapper("tavily", "internet_search", SearchInput, search_operation)


@tool(args_schema=SearchInput)
def internet_search(query: str, topic: str = "general", max_results: int = 5, include_raw_content: bool = False) -> dict:
    """搜索公开网页。返回ToolExecutionResult；degraded不是事实证据。"""
    return search_wrapper.execute(locals()).model_dump(mode="json")


bind_async(internet_search, search_wrapper)
