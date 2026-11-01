"""Check shared model configuration without making API requests."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llm import DEFAULT_MODEL, OPENROUTER_BASE_URL, create_model, create_workflow_model


class SharedModelTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.env_path = Path(temporary.name) / ".env"
        self.env_path.write_text(
            "OPENROUTER_API_KEY=offline-file-key\nLLM_MODEL=test/file-model\n"
        )

    def test_loads_model_and_key_from_the_requested_env_file(self):
        with patch.dict(os.environ, {}, clear=True):
            model = create_model(env_path=self.env_path)

        self.addCleanup(model.root_client.close)
        self.assertEqual(model.model_name, "test/file-model")
        self.assertEqual(model.openai_api_base, OPENROUTER_BASE_URL)
        self.assertEqual(model.openai_api_key.get_secret_value(), "offline-file-key")

    def test_keeps_existing_environment_values_and_uses_the_default_model(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "offline-shell-key"}, clear=True):
            self.env_path.write_text("OPENROUTER_API_KEY=offline-file-key\n")
            model = create_model(env_path=self.env_path)

        self.addCleanup(model.root_client.close)
        self.assertEqual(model.model_name, DEFAULT_MODEL)
        self.assertEqual(model.openai_api_key.get_secret_value(), "offline-shell-key")
        self.assertIsNone(model.temperature)

    def test_workflow_model_keeps_bounded_client_settings(self):
        with patch.dict(os.environ, {}, clear=True):
            model = create_workflow_model(env_path=self.env_path)

        self.addCleanup(model.root_client.close)
        self.assertEqual(model.temperature, 0)
        self.assertEqual(model.max_retries, 1)
        self.assertEqual(model.request_timeout, 45)

    def test_reports_a_missing_openrouter_key_before_constructing_a_model(self):
        self.env_path.write_text("")
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
                create_model(env_path=self.env_path)


if __name__ == "__main__":
    unittest.main()
