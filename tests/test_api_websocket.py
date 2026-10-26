import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.api import server
from app.api.context import reset_session_context, set_session_context, set_thread_context
from app.api.monitor import ConnectionManager, monitor
from app.harness.runtime import active_runs, run_context
from app.tools.markdown_tools import generate_markdown


class APIWebSocketTests(unittest.TestCase):
    def test_start_events_heartbeat_upload_files_download_and_cancel(self):
        async def fake(query, thread_id, run_id=None):
            token = run_context.set(active_runs[thread_id])
            a = set_session_context("unused")
            b = set_thread_context(thread_id)
            try:
                monitor._emit("tool_degraded", "source offline", {"cost_ms": 12.0, "error": {"code": "TimeoutError"}})
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                monitor.report_task_cancelled()
                raise
            finally:
                reset_session_context(a, b)
                run_context.reset(token)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, uploads = root / "output", root / "uploads"
            output.mkdir()
            uploads.mkdir()
            folder = output / "session_api"
            folder.mkdir()
            (folder / "report.md").write_text("# Result", encoding="utf-8")
            with patch.object(server, "output_dir", output), patch.object(server, "updated_dir", uploads), patch.object(server, "run_deep_agent", side_effect=fake):
                with TestClient(server.app) as client:
                    with client.websocket_connect("/ws/api") as socket:
                        socket.send_text("ping")
                        self.assertEqual(socket.receive_json()["type"], "pong")
                        response = client.post("/api/task", json={"query": "test", "thread_id": "api"})
                        self.assertEqual(response.status_code, 200)
                        run_id = response.json()["run_id"]
                        event = socket.receive_json()
                        self.assertEqual(event["event"], "tool_degraded")
                        self.assertEqual(event["data"]["run_id"], run_id)
                        self.assertEqual(event["data"]["cost_ms"], 12)
                        upload = client.post("/api/upload", data={"thread_id": "api"}, files={"files": ("note.txt", b"attachment")})
                        self.assertEqual(upload.json()["status"], "uploaded")
                        self.assertTrue((uploads / "session_api" / "note.txt").exists())
                        listed = client.get("/api/files", params={"path": str(folder)}).json()
                        self.assertEqual(listed["files"][0]["name"], "report.md")
                        download = client.get("/api/download", params={"path": listed["files"][0]["path"]})
                        self.assertEqual(download.text, "# Result")
                        self.assertEqual(client.post("/api/task/api/cancel").json()["status"], "cancelled")
                        self.assertEqual(socket.receive_json()["event"], "task_cancelled")
                    self.assertEqual(client.post("/api/task", json={"query": "x", "thread_id": "../escape"}).status_code, 422)
                    self.assertIn("error", client.get("/api/download", params={"path": str(root / "outside")}).json())
                    self.assertEqual(client.post("/api/upload", data={"thread_id": "api"}, files={"files": ("../bad.txt", b"bad")}).status_code, 422)

    def test_actual_markdown_tool_keeps_session_directory(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b = set_session_context(folder), set_thread_context("md")
            try:
                result = generate_markdown.invoke({"content": "# Evidence\nGrounded report", "filename": "report"})
                self.assertTrue((Path(folder) / "report.md").is_file())
                self.assertIn("Grounded report", (Path(folder) / "report.md").read_text(encoding="utf-8"))
            finally:
                reset_session_context(a, b)
