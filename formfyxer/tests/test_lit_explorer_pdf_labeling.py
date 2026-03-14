import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from formfyxer.lit_explorer import (
    rename_pdf_fields_with_context,
    text_complete,
)


class TestLitExplorerPdfLabeling(unittest.TestCase):
    @patch("formfyxer.lit_explorer.OpenAI")
    def test_text_complete_uses_custom_openai_base_url(self, mock_openai):
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
        )
        mock_openai.return_value = mock_client

        result = text_complete(
            system_message="Respond with JSON only.",
            user_message="{}",
            api_key="sk-test",
            model="gpt-4o-mini",
            openai_base_url="https://example.com/v1",
        )

        self.assertEqual(result, {"ok": True})
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            organization=None,
            base_url="https://example.com/v1",
        )
        self.assertEqual(
            mock_client.chat.completions.create.call_args.kwargs["model"], "gpt-4o-mini"
        )

    @patch("formfyxer.lit_explorer.text_complete")
    @patch(
        "formfyxer.lit_explorer._load_prompt", return_value="Respond with JSON only."
    )
    @patch("formfyxer.lit_explorer.get_original_text_with_fields")
    def test_rename_pdf_fields_with_context_forwards_model_and_base_url(
        self, mock_get_text, mock_load_prompt, mock_text_complete
    ):
        mock_get_text.side_effect = lambda pdf_path, output_path: Path(
            output_path
        ).write_text("Name: {{ field_a }}", encoding="utf-8")
        mock_text_complete.return_value = {"field_mappings": {"field_a": "users1_name"}}

        result = rename_pdf_fields_with_context(
            "fake.pdf",
            ["field_a"],
            api_key="sk-test",
            model="gpt-4o-mini",
            openai_base_url="https://endpoint.example/v1",
        )

        self.assertEqual(result, {"field_a": "users1_name"})
        self.assertEqual(
            mock_text_complete.call_args.kwargs["openai_base_url"],
            "https://endpoint.example/v1",
        )
        self.assertEqual(mock_text_complete.call_args.kwargs["model"], "gpt-4o-mini")


if __name__ == "__main__":
    unittest.main()
