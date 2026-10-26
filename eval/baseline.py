"""Tool-boundary replay of source commit 18e0552 (not original LLM behavior).

Original internet_search: no catch.
Original db_tools: catches mysql.connector.Error and returns Chinese error text.
Original ragflow_tools: catches Exception and returns Chinese error text.
The common benchmark grader never accepts error strings as evidence.
"""
from mysql.connector import Error as MySQLError

BASELINE_COMMIT = "18e0552a17f4caaf290767a7de6f4afba00c7c88"


def legacy_provider_call(source, operation):
    if source == "mysql":
        try:
            return operation()
        except MySQLError as exc:
            return f"查询出现异常：{str(exc)}"
    if source == "ragflow":
        try:
            return operation()
        except Exception as exc:
            return f"提问失败，错误原因：{str(exc)}"
    return operation()


def evaluate_baseline(case, operations, unprotected=False):
    successful_sources = set()
    error_codes = []
    for source in case["sources"]:
        try:
            value = operations[source]() if unprotected else legacy_provider_call(source, operations[source])
        except Exception as exc:
            error_codes.append(type(exc).__name__)
            return {"completed": False, "sources": sorted(successful_sources), "errors": error_codes}
        if isinstance(value, tuple) and value[1]:
            successful_sources.add(source)
        else:
            error_codes.append("legacy_error_string")
    return {"completed": len(successful_sources) >= case["minimum_sources"],
            "sources": sorted(successful_sources), "errors": error_codes}
