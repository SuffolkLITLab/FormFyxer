"""Unit tests for passive voice detection with mocked OpenAI responses."""

import json
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from formfyxer.passive_voice_detection import detect_passive_voice_segments


class TestPassiveVoiceDetection(unittest.TestCase):
    """Test passive voice detection with mocked OpenAI API responses."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        
    def _create_mock_response(self, results_data):
        """Create a mock OpenAI response with the given results data.
        
        Args:
            results_data: List of dicts with 'sentence' and 'fragments' keys
            
        Returns:
            Mock response object that mimics OpenAI's response structure
        """
        mock_response = Mock()
        mock_response.output_text = json.dumps({"results": results_data})
        return mock_response

    def test_detect_passive_voice_basic_passive(self):
        """Test detection of basic passive voice constructions."""
        # Mock response for passive voice sentence
        mock_response = self._create_mock_response([
            {
                "sentence": "The ball was thrown by John.",
                "fragments": ["was thrown"]
            }
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(
            "The ball was thrown by John.",
            openai_client=self.mock_client
        )
        
        self.assertEqual(len(result), 1)
        sentence, fragments = result[0]
        self.assertEqual(sentence, "The ball was thrown by John.")
        self.assertEqual(fragments, ["was thrown"])
        
        # Verify the API was called correctly
        self.mock_client.responses.create.assert_called_once()
        call_args = self.mock_client.responses.create.call_args
        self.assertEqual(call_args[1]['model'], 'gpt-5-nano')

    def test_detect_passive_voice_active_sentence(self):
        """Test that active voice sentences return empty fragments."""
        # Mock response for active voice sentence
        mock_response = self._create_mock_response([
            {
                "sentence": "John threw the ball.",
                "fragments": []
            }
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(
            "John threw the ball.",
            openai_client=self.mock_client
        )
        
        self.assertEqual(len(result), 1)
        sentence, fragments = result[0]
        self.assertEqual(sentence, "John threw the ball.")
        self.assertEqual(fragments, [])

    def test_detect_passive_voice_multiple_sentences(self):
        """Test detection with multiple sentences."""
        # Mock response for multiple sentences
        mock_response = self._create_mock_response([
            {
                "sentence": "The report was completed yesterday.",
                "fragments": ["was completed"]
            },
            {
                "sentence": "The team finished the project.",
                "fragments": []
            },
            {
                "sentence": "The documents were reviewed by the manager.",
                "fragments": ["were reviewed"]
            }
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        text = "The report was completed yesterday. The team finished the project. The documents were reviewed by the manager."
        result = detect_passive_voice_segments(text, openai_client=self.mock_client)
        
        self.assertEqual(len(result), 3)
        
        # Check first sentence (passive)
        sentence1, fragments1 = result[0]
        self.assertEqual(sentence1, "The report was completed yesterday.")
        self.assertEqual(fragments1, ["was completed"])
        
        # Check second sentence (active)
        sentence2, fragments2 = result[1]
        self.assertEqual(sentence2, "The team finished the project.")
        self.assertEqual(fragments2, [])
        
        # Check third sentence (passive)
        sentence3, fragments3 = result[2]
        self.assertEqual(sentence3, "The documents were reviewed by the manager.")
        self.assertEqual(fragments3, ["were reviewed"])

    def test_detect_passive_voice_edge_cases(self):
        """Test edge cases that should not be flagged as passive."""
        # Mock response for edge cases that are not passive
        mock_response = self._create_mock_response([
            {
                "sentence": "Business is very well known career field.",
                "fragments": []
            },
            {
                "sentence": "I am immersing myself in my culture.",
                "fragments": []
            },
            {
                "sentence": "The president was satisfied with the results.",
                "fragments": []
            }
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        sentences = [
            "Business is very well known career field.",
            "I am immersing myself in my culture.",
            "The president was satisfied with the results."
        ]
        
        result = detect_passive_voice_segments(sentences, openai_client=self.mock_client)
        
        self.assertEqual(len(result), 3)
        for sentence, fragments in result:
            self.assertEqual(fragments, [], f"Expected no fragments for: {sentence}")

    def test_detect_passive_voice_sequence_input(self):
        """Test that sequence input is handled correctly."""
        sentences = [
            "The cake was baked this morning.",
            "She baked the cake this morning."
        ]
        
        mock_response = self._create_mock_response([
            {
                "sentence": "The cake was baked this morning.",
                "fragments": ["was baked"]
            },
            {
                "sentence": "She baked the cake this morning.",
                "fragments": []
            }
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(sentences, openai_client=self.mock_client)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][1], ["was baked"])  # First should have fragments
        self.assertEqual(result[1][1], [])  # Second should have no fragments

    def test_detect_passive_voice_invalid_json_response(self):
        """Test handling of invalid JSON responses."""
        # Mock response with invalid JSON
        mock_response = Mock()
        mock_response.output_text = "Invalid JSON response"
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(
            "The ball was thrown.",
            openai_client=self.mock_client
        )
        
        # Should return empty fragments for all sentences when JSON is invalid
        self.assertEqual(len(result), 1)
        sentence, fragments = result[0]
        self.assertEqual(sentence, "The ball was thrown.")
        self.assertEqual(fragments, [])

    def test_detect_passive_voice_empty_response(self):
        """Test handling of empty responses."""
        # Mock response with empty content
        mock_response = Mock()
        mock_response.output_text = ""
        mock_response.output = []
        mock_response.data = []
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(
            "The ball was thrown.",
            openai_client=self.mock_client
        )
        
        # Should return empty fragments for all sentences when response is empty
        self.assertEqual(len(result), 1)
        sentence, fragments = result[0]
        self.assertEqual(sentence, "The ball was thrown.")
        self.assertEqual(fragments, [])

    def test_detect_passive_voice_partial_json_response(self):
        """Test handling of partial JSON in response with extra text."""
        # Mock response with JSON embedded in extra text
        mock_response = Mock()
        mock_response.output_text = 'Here is the analysis: {"results": [{"sentence": "The door was opened.", "fragments": ["was opened"]}]} Hope this helps!'
        
        self.mock_client.responses.create.return_value = mock_response
        
        result = detect_passive_voice_segments(
            "The door was opened.",
            openai_client=self.mock_client
        )
        
        self.assertEqual(len(result), 1)
        sentence, fragments = result[0]
        self.assertEqual(sentence, "The door was opened.")
        self.assertEqual(fragments, ["was opened"])

    def test_detect_passive_voice_short_sentences_filtered(self):
        """Test that sentences with 2 or fewer words are filtered out."""
        with self.assertRaises(ValueError) as context:
            detect_passive_voice_segments("Hi there.", openai_client=self.mock_client)
        
        self.assertIn("no sentences over 2 words", str(context.exception))

    def test_detect_passive_voice_custom_model(self):
        """Test that custom model parameter is passed correctly."""
        mock_response = self._create_mock_response([
            {"sentence": "Test sentence here.", "fragments": []}
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        detect_passive_voice_segments(
            "Test sentence here.",
            openai_client=self.mock_client,
            model="gpt-4"
        )
        
        # Verify the custom model was used
        call_args = self.mock_client.responses.create.call_args
        self.assertEqual(call_args[1]['model'], 'gpt-4')

    @patch('formfyxer.passive_voice_detection._load_prompt')
    def test_prompt_loading(self, mock_load_prompt):
        """Test that the prompt is loaded correctly."""
        mock_load_prompt.return_value = "Test prompt content"
        
        mock_response = self._create_mock_response([
            {"sentence": "Test sentence with more words.", "fragments": []}
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        detect_passive_voice_segments(
            "Test sentence with more words.",
            openai_client=self.mock_client
        )
        
        # Verify prompt was loaded
        mock_load_prompt.assert_called_once()
        
        # Verify the loaded prompt was used in the API call
        call_args = self.mock_client.responses.create.call_args
        input_items = call_args[1]['input']
        system_message = input_items[0]
        self.assertEqual(system_message['content'][0]['text'], "Test prompt content")

    def test_sentence_splitting(self):
        """Test that text is properly split into sentences."""
        mock_response = self._create_mock_response([
            {"sentence": "First sentence here.", "fragments": []},
            {"sentence": "Second sentence here.", "fragments": []},
            {"sentence": "Third sentence here!", "fragments": []}
        ])
        
        self.mock_client.responses.create.return_value = mock_response
        
        text = "First sentence here. Second sentence here. Third sentence here!"
        result = detect_passive_voice_segments(text, openai_client=self.mock_client)
        
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0][0], "First sentence here.")
        self.assertEqual(result[1][0], "Second sentence here.")
        self.assertEqual(result[2][0], "Third sentence here!")


if __name__ == '__main__':
    unittest.main()
