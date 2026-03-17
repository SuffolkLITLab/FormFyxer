import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from formfyxer.lit_explorer import (
    parse_form,
    rename_pdf_fields_with_context,
    text_complete,
)
from formfyxer.pdf_wrangling import get_original_text_with_fields


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

    def test_parse_form_accepts_model_and_openai_base_url(self):
        signature = inspect.signature(parse_form)
        self.assertIn("model", signature.parameters)
        self.assertIn("openai_base_url", signature.parameters)

    def test_get_original_text_with_fields_handles_pdfminer_state(self):
        fixture = Path(__file__).parent / "affidavit_supplement.pdf"
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_path = Path(temp_file.name)
        try:
            get_original_text_with_fields(str(fixture), str(temp_path))
            output = temp_path.read_text(encoding="utf-8")
            self.assertTrue(len(output) > 0)
        finally:
            temp_path.unlink(missing_ok=True)

    @patch("formfyxer.lit_explorer.text_complete")
    @patch("formfyxer.lit_explorer.extract_text")
    @patch("formfyxer.lit_explorer.subprocess.run")
    @patch(
        "formfyxer.lit_explorer._load_prompt", return_value="Respond with JSON only."
    )
    @patch("formfyxer.lit_explorer.get_original_text_with_fields")
    def test_rename_pdf_fields_with_context_uses_ocr_text_when_marker_context_missing(
        self,
        mock_get_text,
        mock_load_prompt,
        mock_subprocess_run,
        mock_extract_text,
        mock_text_complete,
    ):
        mock_get_text.side_effect = lambda pdf_path, output_path: Path(
            output_path
        ).write_text("{{ field_a }}", encoding="utf-8")
        mock_subprocess_run.return_value = SimpleNamespace(returncode=0)
        mock_extract_text.return_value = "Applicant name and address on the form."
        mock_text_complete.return_value = {"field_mappings": {"field_a": "users1_name"}}

        result = rename_pdf_fields_with_context(
            "fake.pdf",
            ["field_a"],
            api_key="sk-test",
            model="gpt-4o-mini",
        )

        self.assertEqual(result, {"field_a": "users1_name"})
        self.assertIn(
            "OCR text from the PDF form",
            mock_text_complete.call_args.kwargs["user_message"],
        )

    @patch("formfyxer.lit_explorer.text_complete")
    @patch("formfyxer.lit_explorer.extract_text", return_value="")
    @patch("formfyxer.lit_explorer.subprocess.run")
    @patch(
        "formfyxer.lit_explorer._load_prompt", return_value="Respond with JSON only."
    )
    @patch("formfyxer.lit_explorer.get_original_text_with_fields")
    def test_rename_pdf_fields_with_context_uses_llm_field_list_fallback_before_regex(
        self,
        mock_get_text,
        mock_load_prompt,
        mock_subprocess_run,
        mock_extract_text,
        mock_text_complete,
    ):
        mock_get_text.side_effect = lambda pdf_path, output_path: Path(
            output_path
        ).write_text("{{ field_a }}", encoding="utf-8")
        mock_subprocess_run.return_value = SimpleNamespace(returncode=1)
        mock_text_complete.return_value = {"field_mappings": {"field_a": "users1_name"}}

        result = rename_pdf_fields_with_context(
            "fake.pdf",
            ["field_a"],
            api_key="sk-test",
            model="gpt-4o-mini",
        )

        self.assertEqual(result, {"field_a": "users1_name"})
        self.assertIn(
            "field names extracted from the PDF",
            mock_text_complete.call_args.kwargs["user_message"],
        )


if __name__ == "__main__":
    unittest.main()
