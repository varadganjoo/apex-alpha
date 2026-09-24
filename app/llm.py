"""Gemini LLM Client & Structured Output Engine for Apex-Alpha.
Uses the google-genai SDK with streaming and Pydantic structured output.
"""

from __future__ import annotations

import logging
import os
from typing import Iterator, Type, TypeVar
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger("apex_alpha.llm")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

T = TypeVar("T", bound=BaseModel)


def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from google import genai
        from google.genai import types

        # Retry 503 load-shedding only; 429 means the free-tier RPM quota is spent and retrying just burns more of it.
        retry = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=10.0, http_status_codes=[503])
        return genai.Client(api_key=api_key, http_options=types.HttpOptions(retry_options=retry))
    except Exception as exc:
        logger.warning(f"Failed to initialize google-genai client: {exc}")
        return None


def generate_structured(
    prompt: str,
    system_instruction: str,
    schema: Type[T],
    model: str = DEFAULT_MODEL,
) -> T:
    """Generates structured Pydantic output using Gemini.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured or generation fails.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable adversarial debate."
        )

    try:
        from google.genai import types

        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.2,
            ),
        )
        parsed: T = response.parsed
        return parsed
    except Exception as exc:
        logger.error(f"Gemini API structured generation failed: {exc}")
        raise RuntimeError(f"Gemini API structured generation failed: {exc}") from exc


def stream_text(
    prompt: str,
    system_instruction: str = "",
    model: str = DEFAULT_MODEL,
) -> Iterator[str]:
    """Streams text generation chunks from Gemini.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not configured or generation fails.
    """
    client = get_gemini_client()
    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please configure GEMINI_API_KEY in .env to enable streaming debate."
        )

    try:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
        ) if system_instruction else types.GenerateContentConfig(temperature=0.3)

        response = client.models.generate_content_stream(
            model=model,
            contents=prompt,
            config=config,
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        logger.error(f"Gemini streaming generation failed: {exc}")
        raise RuntimeError(f"Gemini streaming generation failed: {exc}") from exc
