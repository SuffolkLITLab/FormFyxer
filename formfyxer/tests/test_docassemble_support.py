"""Unit tests for docassemble configuration support utilities."""

import os
import unittest
from unittest.mock import patch

from formfyxer.docassemble_support import get_openai_base_url


class TestDocassembleSupport(unittest.TestCase):
    """Tests for docassemble config-based OpenAI configuration helpers."""

    def test_get_openai_base_url_prefers_explicit(self):
        self.assertEqual(
            get_openai_base_url("https://custom.example/v1"),
            "https://custom.example/v1",
        )

    def test_get_openai_base_url_from_docassemble_direct(self):
        def fake_get_config(key, default=None):
            if key == "openai base url":
                return "https://da.example/v1"
            return default

        with patch("formfyxer.docassemble_support._DOCASSEMBLE_AVAILABLE", True), patch(
            "formfyxer.docassemble_support._da_get_config", fake_get_config
        ):
            self.assertEqual(
                get_openai_base_url(),
                "https://da.example/v1",
            )

    def test_get_openai_base_url_from_docassemble_nested(self):
        def fake_get_config(key, default=None):
            if key == "open ai":
                return {"base url": "https://da.example/v2"}
            return default

        with patch("formfyxer.docassemble_support._DOCASSEMBLE_AVAILABLE", True), patch(
            "formfyxer.docassemble_support._da_get_config", fake_get_config
        ):
            self.assertEqual(
                get_openai_base_url(),
                "https://da.example/v2",
            )

    def test_get_openai_base_url_falls_back_to_env(self):
        with patch(
            "formfyxer.docassemble_support._DOCASSEMBLE_AVAILABLE", False
        ), patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.example/v1"}):
            self.assertEqual(
                get_openai_base_url(),
                "https://env.example/v1",
            )
