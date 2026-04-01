import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from formfyxer.lit_explorer import (
    _rewrite_pdf_fields_in_place,
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

    @patch("formfyxer.lit_explorer.needs_calculations", return_value=False)
    @patch("formfyxer.lit_explorer.is_tagged", return_value=False)
    @patch("formfyxer.lit_explorer.get_sensitive_data_types", return_value=[])
    @patch("formfyxer.lit_explorer.field_types_and_sizes", return_value=[])
    @patch("formfyxer.lit_explorer.all_caps_words", return_value=0)
    @patch("formfyxer.lit_explorer.transformed_sentences", return_value=[])
    @patch("formfyxer.lit_explorer.get_citations", return_value=[])
    @patch("formfyxer.lit_explorer.get_passive_sentences", return_value=[])
    @patch("formfyxer.lit_explorer.split_sentences", return_value=["Sentence one."])
    @patch("formfyxer.lit_explorer.spot", return_value=[])
    @patch("formfyxer.lit_explorer.guess_form_name", return_value="")
    @patch("formfyxer.lit_explorer.describe_form", return_value="")
    @patch("formfyxer.lit_explorer.cleanup_text", side_effect=lambda text: text)
    @patch("formfyxer.lit_explorer.extract_text", return_value="Sample form text")
    @patch(
        "formfyxer.lit_explorer.get_existing_pdf_fields",
        return_value=[
            [
                SimpleNamespace(name="field_a"),
                SimpleNamespace(name="field_b"),
            ]
        ],
    )
    @patch("formfyxer.lit_explorer.unlock_pdf_in_place")
    @patch("formfyxer.lit_explorer.pikepdf.open")
    @patch(
        "formfyxer.lit_explorer.get_openai_api_key_from_sources", return_value="sk-test"
    )
    @patch(
        "formfyxer.lit_explorer.rename_pdf_fields_with_context",
        return_value={
            "field_a": "*users.name.first",
            "field_b": "users.name.last",
        },
    )
    @patch("formfyxer.lit_explorer._rewrite_pdf_fields_in_place")
    def test_parse_form_rewrite_uses_rename_helper_for_nested_fields(
        self,
        mock_rewrite_pdf_fields,
        _mock_rename_with_context,
        _mock_get_api_key,
        mock_pikepdf_open,
        _mock_unlock,
        _mock_get_existing_fields,
        _mock_extract_text,
        _mock_cleanup_text,
        _mock_describe_form,
        _mock_guess_form_name,
        _mock_spot,
        _mock_split_sentences,
        _mock_get_passive_sentences,
        _mock_get_citations,
        _mock_transformed_sentences,
        _mock_all_caps_words,
        _mock_field_types_and_sizes,
        _mock_get_sensitive_data_types,
        _mock_is_tagged,
        _mock_needs_calculations,
    ):
        fake_pdf = SimpleNamespace(
            pages=[object()], docinfo=SimpleNamespace(Title="Form")
        )
        mock_pikepdf_open.return_value = fake_pdf

        with patch(
            "formfyxer.lit_explorer.textstat.text_standard", return_value=6.0
        ), patch(
            "formfyxer.lit_explorer.textstat.difficult_words_list", return_value=[]
        ), patch(
            "formfyxer.lit_explorer.time_to_answer_form", return_value=[-1, -1]
        ):
            result = parse_form("fake.pdf", rewrite=True)

        mock_rewrite_pdf_fields.assert_called_once_with(
            "fake.pdf",
            ["field_a", "field_b"],
            ["*users.name.first", "users.name.last"],
        )
        self.assertEqual(result["fields_old"], ["field_a", "field_b"])
        self.assertEqual(result["fields"], ["*users.name.first", "users.name.last"])

    @patch("formfyxer.lit_explorer._get_named_parent")
    @patch("formfyxer.lit_explorer._unnest_pdf_fields")
    @patch("formfyxer.lit_explorer.pikepdf.Pdf.open")
    def test_rewrite_pdf_fields_in_place_preserves_order_and_full_flat_names(
        self,
        mock_pdf_open,
        mock_unnest_pdf_fields,
        mock_get_named_parent,
    ):
        named_targets = [
            SimpleNamespace(T="repeat"),
            SimpleNamespace(T="repeat"),
            SimpleNamespace(T="old_leaf"),
            SimpleNamespace(T="plain_old"),
        ]
        flattened_fields = [
            {"var_name": "repeat", "all": object()},
            {"var_name": "repeat", "all": object()},
            {"var_name": "group.old_leaf", "all": object()},
            {"var_name": "plain_old", "all": object()},
        ]
        fake_pdf = SimpleNamespace(
            Root=SimpleNamespace(
                AcroForm=SimpleNamespace(
                    Fields=[object(), object(), object(), object()]
                )
            ),
            save=Mock(),
            close=Mock(),
        )
        mock_pdf_open.return_value = fake_pdf
        mock_unnest_pdf_fields.side_effect = [
            [flattened_fields[0]],
            [flattened_fields[1]],
            [flattened_fields[2]],
            [flattened_fields[3]],
        ]
        mock_get_named_parent.side_effect = named_targets

        _rewrite_pdf_fields_in_place(
            "fake.pdf",
            ["repeat", "repeat", "group.old_leaf", "plain_old"],
            [
                "docket_number",
                "docket_number__2",
                "users.name.first",
                "users.name.last",
            ],
        )

        self.assertEqual(named_targets[0].T, "docket_number")
        self.assertEqual(named_targets[1].T, "docket_number__2")
        self.assertEqual(named_targets[2].T, "first")
        self.assertEqual(named_targets[3].T, "users.name.last")
        fake_pdf.save.assert_called_once_with("fake.pdf")
        fake_pdf.close.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
