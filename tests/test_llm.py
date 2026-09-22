"""Unit tests for Gemini Client error handling and invariants in Apex-Alpha.
Ensures that missing API keys or upstream model failures fail loudly with explicit errors
rather than silently returning canned or ungrounded responses.
"""

from unittest.mock import MagicMock
import pytest
from app.llm import generate_structured, stream_text
from app.schemas import DebateArgument


def test_generate_structured_raises_when_api_key_missing(monkeypatch):
    """Invariant: When GEMINI_API_KEY is not configured, generate_structured must raise RuntimeError."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY environment variable is not configured"):
        generate_structured(
            prompt="Analyze NVDA fundamental thesis",
            system_instruction="Generate Bull Case",
            schema=DebateArgument,
        )


def test_generate_structured_raises_when_gemini_api_fails(monkeypatch):
    """Invariant: When Gemini API raises an exception, generate_structured must fail loudly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-valid-looking-key-12345")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("503 Service Unavailable: Model overloaded")
    monkeypatch.setattr("app.llm.get_gemini_client", lambda: mock_client)

    with pytest.raises(RuntimeError, match="Gemini API structured generation failed: 503 Service Unavailable"):
        generate_structured(
            prompt="Analyze NVDA fundamental thesis",
            system_instruction="Generate Bull Case",
            schema=DebateArgument,
        )


def test_stream_text_raises_when_api_key_missing(monkeypatch):
    """Invariant: When GEMINI_API_KEY is not configured, stream_text must raise RuntimeError."""
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY environment variable is not configured"):
        list(stream_text(prompt="Analyze NVDA", system_instruction="Bull Case"))


def test_stream_text_raises_when_gemini_fails(monkeypatch):
    """Invariant: When Gemini streaming API raises an exception, stream_text must fail loudly."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-valid-looking-key-12345")

    mock_client = MagicMock()
    mock_client.models.generate_content_stream.side_effect = Exception("Connection reset by peer")
    monkeypatch.setattr("app.llm.get_gemini_client", lambda: mock_client)

    with pytest.raises(RuntimeError, match="Gemini streaming generation failed: Connection reset by peer"):
        list(stream_text(prompt="Analyze NVDA", system_instruction="Bull Case"))
