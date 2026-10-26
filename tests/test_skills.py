import unittest
from pathlib import Path
from deepagents import create_deep_agent
from deepagents.middleware.skills import _list_skills_with_errors
from langchain_core.messages import AIMessage, ToolMessage
from app.harness.skill_backend import ReadOnlySkillBackend, make_skill_backend
from tests.helpers import ScriptedModel


class SkillTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1] / "skills"
        self.backend = ReadOnlySkillBackend(self.root)

    def test_native_discovery_and_write_denial(self):
        skills, error = _list_skills_with_errors(self.backend, "/")
        self.assertIsNone(error)
        self.assertEqual({s["name"] for s in skills}, {"competitor-analysis", "tech-survey"})
        self.assertIsNotNone(self.backend.write("/tech-survey/SKILL.md", "overwrite").error)
        self.assertIsNotNone(self.backend.edit("/tech-survey/SKILL.md", "a", "b").error)
        self.assertIsNotNone(self.backend.download_files(["/../.env"])[0].error)

    async def test_native_progressive_skill_loading(self):
        model = ScriptedModel(responses=[AIMessage(content="", tool_calls=[{"name": "read_file", "args": {"file_path": "/skills/competitor-analysis/SKILL.md"}, "id": "skill1", "type": "tool_call"}]), AIMessage(content="outline")])
        agent = create_deep_agent(model=model, backend=make_skill_backend(), skills=["/skills/"])
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "competitor comparison"}]})
        self.assertIn("competitor-analysis", str(model.seen[0][0].content))
        self.assertNotIn("试用价与正式价", str(model.seen[0][0].content))
        message = next(m for m in result["messages"] if isinstance(m, ToolMessage))
        self.assertIn("试用价与正式价", message.content)
