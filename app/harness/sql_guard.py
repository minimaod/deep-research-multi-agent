"""Conservative SELECT/SHOW subset, supplemented with server read-only sessions.

Not a full SQL parser or a substitute for a read-only MySQL account.
Complex syntax outside this subset is rejected rather than guessed safe.
"""
import re
from app.harness.schemas import ErrorCode
from app.harness.tool_wrapper import ToolFailure


def validate_readonly_sql(query):
    if not isinstance(query, str) or not query.strip():
        raise ToolFailure(ErrorCode.INVALID, "empty_sql")
    sql = query.strip().removesuffix(";").strip()
    # Reject comment execution tricks and multiple statements even in literals.
    if any(x in sql for x in (";", "--", "/*", "*/", "#", "\\")):
        raise ToolFailure(ErrorCode.INVALID, "unsupported_sql_syntax")
    if not re.match(r"^(SELECT|SHOW)\b", sql, re.I):
        raise ToolFailure(ErrorCode.INVALID, "read_only_queries_required")
    stripped = re.sub(r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"", "''", sql)
    forbidden = r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|GRANT|REVOKE|CALL|HANDLER|INTO|OUTFILE|DUMPFILE|LOAD_FILE|SLEEP|BENCHMARK|GET_LOCK|RELEASE_LOCK|FOR|LOCK|UNLOCK|SET)\b|@|:="
    if re.search(forbidden, stripped, re.I):
        raise ToolFailure(ErrorCode.INVALID, "unsafe_sql_operation")
    # Explicit function allowlist also blocks stored functions with side effects.
    allowed = {"COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "COALESCE", "IFNULL", "DATE", "YEAR", "MONTH", "DAY", "DATEDIFF", "CURDATE", "NOW", "CAST", "CONCAT", "LOWER", "UPPER", "ABS", "DISTINCT", "IN", "AS", "ON"}
    functions = re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped)
    if any(fn.upper() not in allowed for fn in functions):
        raise ToolFailure(ErrorCode.INVALID, "unsupported_sql_function")
    return sql
