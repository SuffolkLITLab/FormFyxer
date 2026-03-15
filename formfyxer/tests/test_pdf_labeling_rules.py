import unittest
from unittest.mock import patch

import numpy as np

from formfyxer.pdf_wrangling import (
    FormField,
    _is_blank_text_field,
    get_possible_fields,
    improve_names_with_surrounding_text,
)


class DummyTextbox:
    def __init__(self, text: str, height: float = 12) -> None:
        self._text = text
        self.height = height

    def get_text(self) -> str:
        return self._text


class DummyPageImage:
    def __init__(self, height: int = 1000) -> None:
        self.height = height

    def save(self, file_obj, format_name: str) -> None:
        file_obj.write(b"dummy")


class TestPdfLabelingRules(unittest.TestCase):
    def test_improve_names_with_preferred_names(self):
        fields = [[FormField.make_textbox("page_0_field_0", (100, 100, 120, 20), 12)]]
        textboxes = [
            [
                {
                    "textbox": DummyTextbox("Applicant Name"),
                    "bbox": (90, 95, 140, 30),
                }
            ]
        ]
        renamed = improve_names_with_surrounding_text(
            fields, textboxes, preferred_names=["custom applicant name"]
        )

        self.assertEqual(renamed[0][0].name, "custom_applicant_name")

    def test_blank_space_rule_callback_is_used(self):
        img_bin = np.zeros((20, 20), dtype=np.uint8)
        callback_args = []

        def custom_blank_rule(img, bbox, line_height, text_lines):
            callback_args.append((bbox, line_height, text_lines))
            return False

        is_blank = _is_blank_text_field(
            img_bin, (1, 10, 10, 5), 5, [], custom_blank_rule
        )

        self.assertFalse(is_blank)
        self.assertEqual(callback_args, [((1, 10, 10, 5), 5, [])])

    def test_blank_space_rule_clamps_out_of_bounds_slice(self):
        img_bin = np.zeros((20, 20), dtype=np.uint8)
        img_bin[-1, -1] = 1

        is_blank = _is_blank_text_field(img_bin, (0, 4, 10, 12), 12, [])

        self.assertTrue(is_blank)

    def test_get_possible_fields_uses_preferred_names(self):
        with patch(
            "formfyxer.pdf_wrangling.convert_from_path",
            return_value=[DummyPageImage()],
        ), patch(
            "formfyxer.pdf_wrangling.get_possible_checkboxes",
            return_value=[(20, 40, 20, 20)],
        ), patch(
            "formfyxer.pdf_wrangling.get_possible_text_fields",
            return_value=[((50, 100, 120, 30), 12)],
        ):
            fields = get_possible_fields(
                "fake.pdf", textboxes=[[]], preferred_names=["first name", "accept box"]
            )

        self.assertEqual(len(fields), 1)
        self.assertEqual(
            [field.name for field in fields[0]], ["first_name", "accept_box"]
        )


if __name__ == "__main__":
    unittest.main()
