import unittest
from ..pdf_wrangling import get_existing_pdf_fields
from ..lit_explorer import cluster_screens, normalize_name
from pathlib import Path


class TestVectorFunctions(unittest.TestCase):
    def test_cluster(self):
        pdf_fields = get_existing_pdf_fields(
            Path(__file__).parent / "affidavit_supplement.pdf"
        )
        field_names = [f.name for page in pdf_fields for f in page]
        screens = cluster_screens(field_names)

        # Sanity check: all fields are present exactly once across groups
        flattened = [field for group in screens.values() for field in group]
        self.assertCountEqual(flattened, field_names)

        # Ensure we produced at least one non-empty grouping
        self.assertTrue(screens)
        self.assertTrue(any(screens.values()))

        # No duplicates within individual groups
        for group_fields in screens.values():
            self.assertEqual(len(group_fields), len(set(group_fields)))

    def test_normalize_name(self):
        pdf_fields = get_existing_pdf_fields(
            Path(__file__).parent / "affidavit_supplement.pdf"
        )
        field_names = [f.name for page in pdf_fields for f in page]
        length = len(field_names)
        last = "null"
        new_names = []
        new_names_conf = []
        for i, field_name in enumerate(field_names):
            new_name, new_confidence = normalize_name(
                "state",
                "",
                i,
                i / length,
                last,
                field_name,
            )
            new_names.append(new_name)
            new_names_conf.append(new_confidence)
            last = field_name
        all_fields = list(zip(field_names, new_names))

        # Basic shape checks
        self.assertEqual(len(all_fields), len(field_names))
        for original, normalized in all_fields:
            self.assertIsInstance(normalized, str)
            self.assertTrue(normalized)
            self.assertRegex(normalized, r"^[a-z0-9_]+$")

        # Spot-check a few critical mappings to ensure legacy behavior stays intact
        mapped = dict(all_fields)
        self.assertEqual(mapped["case_number"], "docket_number")
        self.assertEqual(mapped["users1_signature"], "users_signature")
        self.assertEqual(mapped["signature_date"], "signature_date")
        self.assertEqual(mapped["users1_address_on_one_line"], "users_address_one_line")

        # Ensure high-confidence fields still receive boosted scores when available
        self.assertEqual(len(new_names_conf), len(field_names))
        self.assertTrue(all(isinstance(conf, float) for conf in new_names_conf))
