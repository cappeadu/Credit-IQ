import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.ai_client import (
    OpenAIClientError,
    OpenAIConfigurationError,
    OpenAIExplanationClient,
    OpenAISettings,
)


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class AiClientTests(unittest.TestCase):
    def test_settings_require_an_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(OpenAIConfigurationError):
                OpenAISettings.from_environment()

    def test_settings_read_key_and_model_from_environment(self):
        with patch.dict(
            os.environ,
            {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "test-model"},
            clear=True,
        ):
            settings = OpenAISettings.from_environment()

        self.assertEqual(settings.api_key, "test-key")
        self.assertEqual(settings.model, "test-model")

    def test_client_sends_grounded_input_without_storing_response(self):
        responses = FakeResponses(SimpleNamespace(output_text="Grounded explanation."))
        client = OpenAIExplanationClient(
            OpenAISettings(api_key="test-key", model="test-model"),
            client=SimpleNamespace(responses=responses),
        )

        result = client.generate_text(
            instructions="Explain only the supplied evidence.",
            input_text='{"probability": 0.7}',
        )

        self.assertEqual(result, "Grounded explanation.")
        self.assertEqual(responses.calls[0]["model"], "test-model")
        self.assertFalse(responses.calls[0]["store"])

    def test_client_wraps_provider_failures(self):
        responses = FakeResponses(error=TimeoutError("timed out"))
        client = OpenAIExplanationClient(
            OpenAISettings(api_key="test-key"),
            client=SimpleNamespace(responses=responses),
        )

        with self.assertRaises(OpenAIClientError):
            client.generate_text(instructions="instructions", input_text="context")


if __name__ == "__main__":
    unittest.main()
