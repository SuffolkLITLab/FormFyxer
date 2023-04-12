import unittest
from typing import Dict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from formfyxer.lit_explorer import substitute_phrases


class TestSubstitutePhrases(unittest.TestCase):
    def test_substitute_phrases(self):
        test_cases = [
            {
                "input_string": "The quick brown fox jumps over the lazy dog.",
                "substitution_phrases": {
                    "quick brown fox": "nimble auburn hare",
                    "lazy dog": "slothful canine"
                },
                "expected_output": "The nimble auburn hare jumps over the slothful canine."
            },
            {
                "input_string": "The woman about town is a woman.",
                "substitution_phrases": {
                    "woman about town": "bon vivant",
                    "woman": "person"
                },
                "expected_output": "The bon vivant is a person."
            },
            {
                "input_string": "This is a test sentence with no substitutions.",
                "substitution_phrases": {
                    "nonexistent phrase": "replacement"
                },
                "expected_output": "This is a test sentence with no substitutions."
            },
            {
                "input_string": "This is an example of a sentence with multiple occurrences of a phrase in the sentence.",
                "substitution_phrases": {
                    "sentence": "statement",
                    "occurrences": "instances"
                },
                "expected_output": "This is an example of a statement with multiple instances of a phrase in the statement."
            }
        ]

        for case in test_cases:
            input_string = case["input_string"]
            substitution_phrases: Dict[str, str] = case["substitution_phrases"]
            expected_output = case["expected_output"]
            result, _ = substitute_phrases(input_string, substitution_phrases)
            self.assertEqual(result, expected_output, f"Expected '{expected_output}', but got '{result}'")

if __name__ == '__main__':
    unittest.main()
