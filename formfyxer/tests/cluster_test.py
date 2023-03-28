import unittest
from ..pdf_wrangling import get_existing_pdf_fields
from ..lit_explorer import cluster_screens, normalize_name
from pathlib import Path
import os


class TestVectorFunctions(unittest.TestCase):
    def test_cluster(self):
        pdf_fields = get_existing_pdf_fields(
            Path(__file__).parent / "affidavit_supplement.pdf"
        )
        field_names = [f.name for page in pdf_fields for f in page]
        screens = cluster_screens(field_names, tools_token=os.getenv("TOOLS_TOKEN"))
        existing_3_4_0_stable_aff = {
            "screen_0": ["court_name", "case_name", "users1_name", "case_number"],
            "screen_1": ["users1_birthdate"],
            "screen_2": [
                "user_income_nonemployment",
                "user_income_gross_annual",
                "user_income_gross_monthly",
                "user_income_after_tax1",
                "user_income_after_tax2",
                "user_income_net",
                "user_debts",
            ],
            "screen_3": [
                "user_tax_federal",
                "user_tax_state",
                "user_tax_social_security",
                "user_tax_medicare",
                "user_tax_other",
                "user_tax_other_amount",
                "user_tax_total",
            ],
            "screen_4": ["user_work_name", "user_househould_member_work_name"],
            "screen_5": [
                "user_dependents_number",
                "user_expense_rent",
                "user_expense_food",
                "user_expense_electricity",
                "user_expense_gas",
                "user_expense_oil",
                "user_expense_water",
                "user_expense_telephone",
                "user_expense_laundry",
                "user_expense_transportation",
                "user_expense_other_amount",
                "user_expense_total",
                "user_expense_other",
                "user_accounts_type",
                "user_property_type",
                "user_accounts_balance",
                "user_property_value",
            ],
            "screen_6": [
                "user_grade_school_completed",
                "user_training",
                "user_disabilities",
                "user_expense_health_insurance",
                "user_expense_uninsured_medical",
                "user_expense_childcare",
                "user_expense_education",
                "user_expense_child_support",
                "user_expense_clothing",
                "user_home_value",
                "user_home_debt",
            ],
            "screen_7": [
                "user_expense_car_insurance",
                "user_owns_car_yes",
                "user_car_make",
                "user_car_value",
                "user_car_debt",
                "user_car_year",
            ],
            "screen_8": ["user_owns_home_yes", "user_owns_home_no", "user_owns_car_no"],
            "screen_9": ["miscellaneous_facts"],
            "screen_10": ["users1_name__2"],
            "screen_11": ["users1_address_on_one_line", "users1_address_line_one"],
            "screen_12": ["users1_address_city", "users1_address_state"],
            "screen_13": ["users1_address_zip"],
            "screen_14": ["signature_date", "users1_signature"],
        }
        self.maxDiff = None
        self.assertDictEqual(existing_3_4_0_stable_aff, screens)

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
                tools_token=os.getenv("TOOLS_TOKEN"),
            )
            new_names.append(new_name)
            new_names_conf.append(new_confidence)
            last = field_name
        all_fields = list(zip(field_names, new_names))
        existing_3_4_0 = [
            ("court_name", "court_name"),
            ("case_name", "case_name"),
            ("users1_name", "*users1_name"),
            ("users1_address_on_one_line", "users_address_one_line"),
            ("users1_birthdate", "*users1_birthdate"),
            ("user_grade_school_completed", "user_grade_school_completed"),
            ("user_training", "user_training"),
            ("user_disabilities", "user_disabilities"),
            ("user_dependents_number", "user_dependents_number"),
            ("user_work_name", "user_work_name"),
            ("user_income_nonemployment", "user_income_nonemployment"),
            ("user_income_gross_annual", "user_income_gross_annual"),
            ("case_number", "*docket_number"),
            ("user_income_gross_monthly", "user_income_gross_monthly"),
            ("user_tax_federal", "user_tax_federal"),
            ("user_tax_state", "user_tax_state"),
            ("user_tax_social_security", "user_tax_social_security"),
            ("user_tax_medicare", "user_tax_medicare"),
            ("user_tax_other", "user_tax"),
            ("user_tax_other_amount", "user_tax_amount"),
            ("user_tax_total", "user_tax_total"),
            ("user_income_after_tax1", "user_income_tax"),
            ("user_househould_member_work_name", "user_member_work_name"),
            ("user_income_after_tax2", "user_income_tax"),
            ("user_expense_rent", "user_expense_rent"),
            ("user_expense_food", "user_expense_food"),
            ("user_expense_electricity", "user_expense_electricity"),
            ("user_expense_gas", "user_expense_gas"),
            ("user_expense_oil", "user_expense_oil"),
            ("user_expense_water", "user_expense_water"),
            ("user_expense_telephone", "user_expense_telephone"),
            ("user_expense_health_insurance", "expense_health_insurance"),
            ("user_expense_uninsured_medical", "expense_uninsured_medical"),
            ("user_expense_childcare", "user_expense_childcare"),
            ("user_expense_education", "user_expense_education"),
            ("user_expense_child_support", "user_expense_child_support"),
            ("user_expense_clothing", "user_expense_clothing"),
            ("user_expense_laundry", "user_expense_laundry"),
            ("user_expense_car_insurance", "user_expense_car_insurance"),
            ("user_expense_transportation", "user_expense_transportation"),
            ("user_expense_other_amount", "user_expense_amount"),
            ("user_expense_total", "user_expense_total"),
            ("user_income_net", "user_income_net"),
            ("user_expense_other", "user_expense"),
            ("user_owns_home_yes", "user_owns_home_yes"),
            ("user_owns_home_no", "user_owns_home"),
            ("user_home_value", "user_home_value"),
            ("user_home_debt", "user_home_debt"),
            ("user_owns_car_yes", "user_owns_car_yes"),
            ("user_owns_car_no", "user_owns_car"),
            ("user_car_make", "user_car_make"),
            ("user_car_value", "user_car_value"),
            ("user_car_debt", "user_car_debt"),
            ("user_accounts_type", "user_accounts_type"),
            ("user_property_type", "user_property_type"),
            ("user_debts", "user_debts"),
            ("miscellaneous_facts", "miscellaneous_facts"),
            ("users1_name__2", "users_name"),
            ("users1_address_line_one", "*users1_address_line_one"),
            ("users1_address_city", "*users1_address_city"),
            ("users1_address_state", "*users1_address_state"),
            ("users1_address_zip", "*users1_address_zip"),
            ("signature_date", "*signature_date"),
            ("user_accounts_balance", "user_accounts_balance"),
            ("user_car_year", "user_car_year"),
            ("user_property_value", "user_property_value"),
            ("users1_signature", "*users1_signature"),
        ]
        self.maxDiff = None
        self.assertListEqual(existing_3_4_0, all_fields)
