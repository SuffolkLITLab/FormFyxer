import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile
from unittest.mock import patch

import numpy as np
import pikepdf
from reportlab.pdfgen import canvas

from formfyxer.pdf_wrangling import (
    FormField,
    _clamp_rect_to_page,
    _is_blank_text_field,
    _extract_unique_text_anchors_from_page,
    _local_anchor_residual_for_point,
    _normalize_anchor_text,
    get_existing_pdf_fields,
    get_possible_fields,
    improve_names_with_surrounding_text,
    copy_pdf_fields,
    set_fields,
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


class DummyPage:
    def __init__(self, width: float = 100.0, height: float = 80.0) -> None:
        self.MediaBox = [0.0, 0.0, width, height]


class TestPdfLabelingRules(unittest.TestCase):
    def test_anchor_text_normalization(self):
        self.assertEqual(
            _normalize_anchor_text("  Applicant Name:  "),
            "applicant name",
        )

    def test_extract_unique_text_anchors_filters_duplicates(self):
        textboxes = [
            {"textbox": DummyTextbox("Applicant Name"), "bbox": (100, 100, 80, 10)},
            {"textbox": DummyTextbox("Address"), "bbox": (100, 80, 60, 10)},
            {"textbox": DummyTextbox("Address"), "bbox": (300, 80, 60, 10)},
        ]
        anchors = _extract_unique_text_anchors_from_page(textboxes)
        self.assertIn("applicant name", anchors)
        self.assertNotIn("address", anchors)

    def test_local_anchor_residual_uses_nearest_anchor(self):
        transform = {
            "scale_x": 1.0,
            "scale_y": 1.0,
            "shift_x": 0.0,
            "shift_y": 0.0,
            "matched_anchor_pairs": [((100.0, 100.0), (130.0, 90.0))],
        }
        residual_x, residual_y = _local_anchor_residual_for_point(
            (110.0, 105.0), transform, max_distance=100.0
        )
        self.assertAlmostEqual(residual_x, 30.0, places=5)
        self.assertAlmostEqual(residual_y, -10.0, places=5)

    def test_clamp_rect_to_page_handles_oversized_rect(self):
        page = DummyPage(width=100.0, height=80.0)
        clamped = _clamp_rect_to_page((-20.0, -10.0, 140.0, 120.0), page)
        self.assertEqual(clamped, (0.0, 0.0, 100.0, 80.0))

    @patch("formfyxer.pdf_wrangling._get_page_anchor_transforms")
    def test_copy_pdf_fields_anchor_adjusts_rectangles(self, mock_get_transforms):
        with NamedTemporaryFile(suffix=".pdf", delete=False) as source_blank_tmp:
            source_blank_path = Path(source_blank_tmp.name)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as source_tmp:
            source_path = Path(source_tmp.name)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as destination_tmp:
            destination_path = Path(destination_tmp.name)

        try:
            c = canvas.Canvas(str(source_blank_path))
            c.drawString(72, 720, "Source")
            c.showPage()
            c.save()

            source_fields = [
                [FormField.make_textbox("test_field", (100, 200, 120, 30), 12)]
            ]
            set_fields(
                str(source_blank_path), str(source_path), source_fields, overwrite=True
            )

            c = canvas.Canvas(str(destination_path))
            c.drawString(72, 720, "Destination")
            c.showPage()
            c.save()

            mock_get_transforms.return_value = [
                {
                    "scale_x": 1.0,
                    "scale_y": 1.0,
                    "shift_x": 20.0,
                    "shift_y": 10.0,
                    "matched_anchor_pairs": [],
                }
            ]

            with pikepdf.Pdf.open(str(source_path)) as source_pdf, pikepdf.Pdf.open(
                str(destination_path), allow_overwriting_input=True
            ) as destination_pdf:
                out_pdf = copy_pdf_fields(
                    source_pdf=source_pdf,
                    destination_pdf=destination_pdf,
                    anchor=True,
                )
                rect = [float(v) for v in out_pdf.pages[0].Annots[0].Rect]
                self.assertEqual(rect, [120.0, 210.0, 240.0, 240.0])
        finally:
            source_blank_path.unlink(missing_ok=True)
            source_path.unlink(missing_ok=True)
            destination_path.unlink(missing_ok=True)

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

    def test_get_existing_pdf_fields_infers_page_without_widget_p_pointer(self):
        with NamedTemporaryFile(suffix=".pdf", delete=False) as base_tmp:
            base_path = Path(base_tmp.name)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as labeled_tmp:
            labeled_path = Path(labeled_tmp.name)

        try:
            c = canvas.Canvas(str(base_path))
            c.drawString(72, 720, "Page 1")
            c.showPage()
            c.drawString(72, 720, "Page 2")
            c.showPage()
            c.drawString(72, 720, "Page 3")
            c.showPage()
            c.save()

            fields_per_page = [
                [FormField.make_textbox("field_page_1", (72, 650, 140, 20), 12)],
                [FormField.make_textbox("field_page_2", (72, 650, 140, 20), 12)],
                [FormField.make_textbox("field_page_3", (72, 650, 140, 20), 12)],
            ]
            set_fields(str(base_path), str(labeled_path), fields_per_page, overwrite=True)

            loaded_fields = get_existing_pdf_fields(str(labeled_path))
            self.assertEqual([len(page_fields) for page_fields in loaded_fields], [1, 1, 1])
        finally:
            base_path.unlink(missing_ok=True)
            labeled_path.unlink(missing_ok=True)

    def test_get_existing_pdf_fields_keeps_field_without_p_or_annots_mapping(self):
        with NamedTemporaryFile(suffix=".pdf", delete=False) as base_tmp:
            base_path = Path(base_tmp.name)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as patched_tmp:
            patched_path = Path(patched_tmp.name)

        try:
            c = canvas.Canvas(str(base_path))
            c.drawString(72, 720, "Page 1")
            c.showPage()
            c.drawString(72, 720, "Page 2")
            c.showPage()
            c.save()

            with pikepdf.Pdf.open(str(base_path), allow_overwriting_input=True) as pdf:
                logical_only_field = pikepdf.Dictionary(
                    FT=pikepdf.Name("/Tx"),
                    T=pikepdf.String("logical_only_field"),
                    F=4,
                    Rect=pikepdf.Array([72, 650, 212, 670]),
                )
                pdf.Root.AcroForm = pikepdf.Dictionary(
                    Fields=pikepdf.Array([pdf.make_indirect(logical_only_field)])
                )
                pdf.save(str(patched_path))

            loaded_fields = get_existing_pdf_fields(str(patched_path))
            self.assertEqual(len(loaded_fields), 2)
            self.assertEqual(len(loaded_fields[0]), 1)
            self.assertEqual(len(loaded_fields[1]), 0)
            self.assertEqual(loaded_fields[0][0].name, "logical_only_field")
        finally:
            base_path.unlink(missing_ok=True)
            patched_path.unlink(missing_ok=True)

    def test_get_existing_pdf_fields_uses_overlap_avoidance_for_unmapped_fields(self):
        with NamedTemporaryFile(suffix=".pdf", delete=False) as base_tmp:
            base_path = Path(base_tmp.name)
        with NamedTemporaryFile(suffix=".pdf", delete=False) as patched_tmp:
            patched_path = Path(patched_tmp.name)

        try:
            c = canvas.Canvas(str(base_path))
            c.drawString(72, 720, "Page 1")
            c.showPage()
            c.drawString(72, 720, "Page 2")
            c.showPage()
            c.save()

            with pikepdf.Pdf.open(str(base_path), allow_overwriting_input=True) as pdf:
                field_one = pikepdf.Dictionary(
                    FT=pikepdf.Name("/Tx"),
                    T=pikepdf.String("logical_field_1"),
                    F=4,
                    Rect=pikepdf.Array([72, 650, 212, 670]),
                )
                field_two = pikepdf.Dictionary(
                    FT=pikepdf.Name("/Tx"),
                    T=pikepdf.String("logical_field_2"),
                    F=4,
                    Rect=pikepdf.Array([72, 650, 212, 670]),
                )
                pdf.Root.AcroForm = pikepdf.Dictionary(
                    Fields=pikepdf.Array(
                        [pdf.make_indirect(field_one), pdf.make_indirect(field_two)]
                    )
                )
                pdf.save(str(patched_path))

            loaded_fields = get_existing_pdf_fields(str(patched_path))
            self.assertEqual(len(loaded_fields), 2)
            self.assertEqual([len(page_fields) for page_fields in loaded_fields], [1, 1])
            self.assertEqual(loaded_fields[0][0].name, "logical_field_1")
            self.assertEqual(loaded_fields[1][0].name, "logical_field_2")
        finally:
            base_path.unlink(missing_ok=True)
            patched_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
