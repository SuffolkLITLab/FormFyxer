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


    def test_phrase_and_position_various_orders(self):
        test_cases = [
            ("The quick brown fox jumped over the lazy dog.",
             {"quick brown": "swift reddish", "lazy dog": "sleepy canine"},
            "The swift reddish fox jumped over the sleepy canine.",
             [(4, 17), (38, 51)]),
             
            ("The sun is shining, and the sunshine is bright.",
             {"sun": "moon", "sunshine": "moonlight"},
             "The moon is shining, and the moonlight is bright.",
             [(4, 8), (29, 38)]),
             
            ("The black cat sat on the black mat.",
             {"black cat": "brown dog", "black": "red"},
             "The brown dog sat on the red mat.",
             [(4, 13), (25, 28)]),
             
            ("The woman about town is a woman.",
             {"woman about town": "bon vivant", "woman": "person"},
             "The bon vivant is a person.",
             [(4, 14), (20, 26)]),
             
            ("The fast car raced past the stationary car.",
             {"fast car": "speedy vehicle", "car": "automobile"},
             "The speedy vehicle raced past the stationary automobile.",
             [(4, 18), (45, 55)]),
            ("This is an example sentence for a woman about town to demonstrate the function.",
             {"woman about town": "bon vivant","example": "sample","demonstrate": "illustrate",},
             "This is an sample sentence for a bon vivant to illustrate the function.",
             [(11, 17), (33, 43), (47, 57)],
            ),
        ]

        for input_string, substitution_phrases, expected_output, expected_positions in test_cases:
            new_string, actual_positions = substitute_phrases(input_string, substitution_phrases)
            self.assertEqual(new_string, expected_output)
            self.assertEqual(actual_positions, expected_positions)


if __name__ == '__main__':
    unittest.main()
