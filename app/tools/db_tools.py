"""Read-only MySQL adapters. Existing tool names and argument names stay stable."""
import hashlib
import json
import math
import os
import re
from dotenv import load_dotenv
from langchain_core.tools import tool
from mysql.connector import connect
from app.harness.schemas import EmptyInput, ErrorCode, EvidenceItem, SQLData, SQLInput, TableInput
from app.harness.sql_guard import validate_readonly_sql
from app.harness.tool_wrapper import ToolFailure, ToolWrapper, bind_async
load_dotenv()
MAX_ROWS = 100


def get_db_config(timeout=20):
    try:
        config = {"host": os.getenv("MYSQL_HOST", "localhost"), "port": int(os.getenv("MYSQL_PORT", "3306")),
            "user": os.getenv("MYSQL_USER"), "password": os.getenv("MYSQL_PASSWORD"), "database": os.getenv("MYSQL_DATABASE"),
            "charset": os.getenv("MYSQL_CHARSET", "utf8mb4"), "collation": os.getenv("MYSQL_COLLATION", "utf8mb4_unicode_ci"),
            "autocommit": False, "consume_results": True, "sql_mode": os.getenv("MYSQL_SQL_MODE", "TRADITIONAL"),
            "connection_timeout": max(1, min(5, math.ceil(timeout))),
            "read_timeout": max(1, math.ceil(timeout)), "write_timeout": max(1, math.ceil(timeout))}
    except ValueError as exc:
        raise ToolFailure(ErrorCode.UNAVAILABLE, "invalid_mysql_configuration") from exc
    if any(config[key] is None for key in ("user", "password", "database")):
        raise ToolFailure(ErrorCode.UNAVAILABLE, "missing_mysql_configuration")
    return config


def query_rows(query, timeout, known_table=None):
    query = validate_readonly_sql(query)
    if re.match(r"SELECT\b", query, re.I):
        limits = re.findall(r"\bLIMIT\s+(\d+)(?:\s*,\s*(\d+))?", query, re.I)
        if any(int(offset_or_count) > MAX_ROWS + 1 or (count and int(count) > MAX_ROWS + 1) for offset_or_count, count in limits):
            raise ToolFailure(ErrorCode.INVALID, "row_limit_exceeds_101")
        if not limits:
            query += f" LIMIT {MAX_ROWS + 1}"
    with connect(**get_db_config(timeout)) as conn:
        conn.start_transaction(readonly=True)
        with conn.cursor() as cursor:
            cursor.execute(f"SET SESSION MAX_EXECUTION_TIME={max(1, int(timeout * 1000))}")
            if known_table:
                cursor.execute("SHOW TABLES")
                if known_table not in {row[0] for row in cursor.fetchall()}:
                    raise ToolFailure(ErrorCode.INVALID, "unknown_table")
            cursor.execute(query)
            if not cursor.description:
                raise ToolFailure(ErrorCode.EMPTY, "no_result_set")
            columns = [str(d[0]) for d in cursor.description]
            rows = cursor.fetchmany(MAX_ROWS + 1)
            data = SQLData(columns=columns, rows=[[None if v is None else str(v) for v in row] for row in rows[:MAX_ROWS]], truncated=len(rows) > MAX_ROWS)
    if not data.rows:
        raise ToolFailure(ErrorCode.EMPTY, "no_database_rows")
    evidence = [EvidenceItem(id="sql-" + hashlib.sha256((query + str(i) + json.dumps(row, ensure_ascii=False)).encode()).hexdigest()[:16],
        source="mysql", kind="row", locator="mysql:query:" + hashlib.sha256(query.encode()).hexdigest()[:16],
        # Preserve duplicate column names in joins; dict(zip(...)) would drop cells.
        text=json.dumps({"columns": data.columns, "values": row}, ensure_ascii=False)) for i, row in enumerate(data.rows)]
    return data, evidence, ["rows_limited_to_100"] if data.truncated else []


def list_operation(params, timeout):
    return query_rows("SHOW TABLES", timeout)


def table_operation(params, timeout):
    return query_rows(f"SELECT * FROM `{params.table_name}` LIMIT 100", timeout, params.table_name)


def sql_operation(params, timeout):
    return query_rows(params.query, timeout)


list_wrapper = ToolWrapper("mysql", "list_sql_tables", EmptyInput, list_operation)
table_wrapper = ToolWrapper("mysql", "get_table_data", TableInput, table_operation)
sql_wrapper = ToolWrapper("mysql", "execute_sql_query", SQLInput, sql_operation)


@tool(args_schema=EmptyInput)
def list_sql_tables() -> dict:
    """列出真实可用表。先确认表名再查询；返回ToolExecutionResult。"""
    return list_wrapper.execute({}).model_dump(mode="json")


@tool(args_schema=TableInput)
def get_table_data(table_name: str) -> dict:
    """校验表名后预览前100行，保留字段和NULL；返回结构化结果。"""
    return table_wrapper.execute({"table_name": table_name}).model_dump(mode="json")


@tool(args_schema=SQLInput)
def execute_sql_query(query: str) -> dict:
    """执行只读SELECT/SHOW，禁止修改、文件操作、多语句和未知函数。"""
    return sql_wrapper.execute({"query": query}).model_dump(mode="json")


for _tool, _wrapper in ((list_sql_tables, list_wrapper), (get_table_data, table_wrapper), (execute_sql_query, sql_wrapper)):
    bind_async(_tool, _wrapper)
