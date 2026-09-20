"""Minimal server-side client for OpenAI explanation requests."""

import os
from dataclasses import dataclass
from typing import Any

OPENAI_API_KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"
OPENAI_MODEL_ENVIRONMENT_VARIABLE = "OPENAI_MODEL"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENAI_TIMEOUT_SECONDS = 30.0


class OpenAIConfigurationError(RuntimeError):
    """Raised when the OpenAI client is not configured."""


class OpenAIClientError(RuntimeError):
    """Raised when an OpenAI explanation request fails."""


@dataclass(frozen=True)
class OpenAISettings:
    """Configuration required by the server-side OpenAI client."""

    api_key: str
    model: str = DEFAULT_OPENAI_MODEL
    timeout_seconds: float = DEFAULT_OPENAI_TIMEOUT_SECONDS

    @classmethod
    def from_environment(cls) -> "OpenAISettings":
        api_key = os.getenv(OPENAI_API_KEY_ENVIRONMENT_VARIABLE)
        if not api_key:
            raise OpenAIConfigurationError(
                f"Set {OPENAI_API_KEY_ENVIRONMENT_VARIABLE} before using AI explanations."
            )
        model = os.getenv(OPENAI_MODEL_ENVIRONMENT_VARIABLE, DEFAULT_OPENAI_MODEL)
        return cls(api_key=api_key, model=model)


class OpenAIExplanationClient:
    """Thin wrapper around the OpenAI Responses API.

    The SDK is imported only when the client is constructed, so prediction
    endpoints and unit tests do not require an API key or network access.
    """

    def __init__(self, settings: OpenAISettings | None = None, client: Any = None):
        self.settings = settings or OpenAISettings.from_environment()
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise OpenAIConfigurationError(
                    "The openai package is not installed in the active environment."
                ) from exc
            client = OpenAI(
                api_key=self.settings.api_key,
                timeout=self.settings.timeout_seconds,
                max_retries=0,
            )
        self._client = client

    def generate_text(self, *, instructions: str, input_text: str) -> str:
        """Generate one explanation response from grounded context."""
        try:
            response = self._client.responses.create(
                model=self.settings.model,
                instructions=instructions,
                input=input_text,
                store=False,
            )
        except Exception as exc:
            raise OpenAIClientError("The OpenAI explanation request failed.") from exc

        output_text = getattr(response, "output_text", "")
        if not output_text:
            raise OpenAIClientError("The OpenAI response did not contain text.")
        return output_text
