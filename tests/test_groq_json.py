"""Groq's gpt-oss sometimes double-escapes newlines inside JSON strings; both forms must parse to real newlines."""

from app import llm
from app.llm import generate_structured
from app.schemas import DebateArgument


def test_groq_json_newlines_normal_and_double_escaped(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    raw = r'{"agent_type": "bull", "claim": "a\nb", "thesis": "1. X.\\n2. Y.", "evidence_points": ["e"]}'
    monkeypatch.setattr(llm, "groq_chat", lambda *a, **k: raw)

    arg = generate_structured("p", "s", DebateArgument)

    assert arg.claim == "a\nb"
    assert arg.thesis == "1. X.\n2. Y."
