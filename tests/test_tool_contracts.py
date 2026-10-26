import json
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from pydantic import ValidationError
from app.harness.schemas import AskInput, ErrorCode, SearchInput, SQLData, TableInput, ToolExecutionResult
from app.harness.sql_guard import validate_readonly_sql
from app.harness.tool_wrapper import ToolFailure
from app.tools import db_tools, ragflow_tools, tavily_tool


class ContractTests(unittest.TestCase):
    def test_strict_input_models(self):
        for args in ({"query": "x", "max_results": "3"}, {"query": "x", "extra": True}, {"query": ""}):
            with self.assertRaises(ValidationError):
                SearchInput.model_validate(args)
        with self.assertRaises(ValidationError):
            TableInput(table_name="drugs; DROP TABLE drugs")
        with self.assertRaises(ValidationError):
            AskInput(chat_name="", question="q")

    def test_output_status_and_sql_width(self):
        with self.assertRaises(ValidationError):
            ToolExecutionResult(source="mysql", tool_name="x", call_id="x", status="ok")
        with self.assertRaises(ValidationError):
            SQLData(columns=["a"], rows=[["1", "2"]])

    def test_sql_guard_blocks_side_effects(self):
        for sql in ("DROP TABLE drugs", "SELECT 1; DELETE FROM drugs", "SELECT SLEEP(100)", "SELECT * INTO OUTFILE '/tmp/x' FROM drugs", "SELECT LOAD_FILE('/etc/passwd')", "SELECT unknown_function()", "SELECT @x:=1", "SELECT 1 /*! DROP TABLE drugs */", "SELECT * FROM drugs FOR UPDATE"):
            with self.subTest(sql=sql), self.assertRaises(ToolFailure):
                validate_readonly_sql(sql)
        self.assertEqual(validate_readonly_sql("SELECT COUNT(*) FROM drugs;"), "SELECT COUNT(*) FROM drugs")

    def test_tavily_payload_and_native_timeout(self):
        client = MagicMock()
        client.search.return_value = {"results": [{"title": "Title", "url": "https://example.invalid/a", "content": "Source text", "score": .8}]}
        with patch.object(tavily_tool, "get_tavily_client", return_value=client):
            result = ToolExecutionResult.model_validate(tavily_tool.internet_search.invoke({"query": "test"}))
        self.assertEqual(result.evidence[0].text, "Source text")
        self.assertGreater(client.search.call_args.kwargs["timeout"], 0)
        client.close.assert_called_once()

    def test_missing_config_is_structured_not_import_failure(self):
        with patch.dict("os.environ", {}, clear=True):
            with patch("app.tools.tavily_tool.load_dotenv"):
                result = tavily_tool.search_wrapper.execute({"query": "test"})
        self.assertEqual(result.error.code, ErrorCode.UNAVAILABLE)
        self.assertEqual(result.attempts, 1)

    def test_mysql_json_normalizes_null_and_decimal(self):
        from decimal import Decimal
        conn, cursor = MagicMock(), MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.description = [("amount",), ("note",)]
        cursor.fetchmany.return_value = [(Decimal("1.25"), None)]
        with patch.object(db_tools, "get_db_config", return_value={}), patch.object(db_tools, "connect", return_value=conn):
            result = db_tools.execute_sql_query.invoke({"query": "SELECT amount,note FROM drugs"})
        parsed = ToolExecutionResult.model_validate(result)
        self.assertEqual(parsed.data.rows, [["1.25", None]])
        conn.start_transaction.assert_called_once_with(readonly=True)
        self.assertIn("LIMIT 101", cursor.execute.call_args.args[0])

    def test_mysql_mutation_never_connects(self):
        with patch.object(db_tools, "connect") as connect:
            result = db_tools.execute_sql_query.invoke({"query": "DELETE FROM drugs"})
        connect.assert_not_called()
        self.assertEqual(result["error"]["code"], "InvalidParam")

    def test_join_duplicate_column_names_do_not_lose_cells(self):
        conn, cursor = MagicMock(), MagicMock()
        conn.__enter__.return_value = conn
        conn.cursor.return_value.__enter__.return_value = cursor
        cursor.description = [("id",), ("id",)]
        cursor.fetchmany.return_value = [(1, 2)]
        with patch.object(db_tools, "get_db_config", return_value={}), patch.object(db_tools, "connect", return_value=conn):
            result = db_tools.execute_sql_query.invoke({"query": "SELECT a.id,b.id FROM a JOIN b ON a.id=b.id"})
        row = json.loads(result["evidence"][0]["text"])
        self.assertEqual(row, {"columns": ["id", "id"], "values": ["1", "2"]})

    def test_ragflow_missing_assistant_is_empty(self):
        client = MagicMock()
        client.list_chats.return_value = []
        with patch.object(ragflow_tools, "get_ragflow_client", return_value=client):
            result = ragflow_tools.create_ask_delete.invoke({"chat_name": "missing", "question": "test"})
        self.assertEqual(result["error"]["code"], "EmptyResult")
        client.close.assert_called_once()

    def test_ragflow_references_and_session_cleanup(self):
        client, chat, response = MagicMock(), MagicMock(), MagicMock()
        client.list_chats.return_value = [chat]
        chat.create_session.return_value = SimpleNamespace(id="session1")
        client.post.return_value = response
        chunk = {"data": {"answer": "Answer", "reference": {"chunks": [{"id": "c1", "document_id": "d1", "document_name": "Report", "content": "Original evidence"}]}}}
        response.iter_lines.return_value = ["data:" + json.dumps(chunk), "data:[DONE]"]
        with patch.object(ragflow_tools, "get_ragflow_client", return_value=client):
            result = ToolExecutionResult.model_validate(ragflow_tools.create_ask_delete.invoke({"chat_name": "report", "question": "test"}))
        self.assertFalse(result.data.references_missing)
        self.assertEqual(result.evidence[0].text, "Original evidence")
        chat.delete_sessions.assert_called_once_with(ids=["session1"])
        response.close.assert_called_once()

    def test_ragflow_cleanup_on_stream_failure(self):
        client, chat, response = MagicMock(), MagicMock(), MagicMock()
        client.list_chats.return_value = [chat]
        client.post.return_value = response
        response.iter_lines.side_effect = TimeoutError()
        with patch.object(ragflow_tools, "get_ragflow_client", return_value=client):
            result = ragflow_tools.create_ask_delete.invoke({"chat_name": "report", "question": "test"})
        self.assertEqual(result["error"]["code"], "TimeoutError")
        chat.delete_sessions.assert_called_once()
        response.close.assert_called_once()

    def test_ragflow_http_timeout_is_explicit(self):
        from app.ragflow.rag_config import get_ragflow_client
        with patch("app.ragflow.rag_config._load_ragflow_env", return_value=("fake", "http://example.invalid")), patch("requests.Session") as session:
            client = get_ragflow_client(.1)
            client.get("/chats")
            self.assertTrue(session.return_value.request.call_args.kwargs["timeout"])
            client.close()
