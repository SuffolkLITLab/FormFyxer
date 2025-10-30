"""Passive voice detection utilities backed by OpenAI's ChatCompletion API."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai import AuthenticationError

from .docassemble_support import get_openai_api_key

__all__ = ["detect_passive_voice_segments", "split_sentences"]

load_dotenv()

DEFAULT_MODEL = "gpt-5-nano"

_cached_client: Optional[OpenAI] = None
_api_key = os.getenv("OPENAI_API_KEY")
_organization = os.getenv("OPENAI_ORGANIZATION") or os.getenv("OPENAI_ORG")


def _load_prompt() -> str:
    """Load the passive voice detection prompt from the prompts directory.

    Returns:
        The prompt text as a string.

    Raises:
        FileNotFoundError: If the prompt file cannot be found.
    """
    # Get the path to the prompts directory relative to this module
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "prompts" / "passive_voice.txt"

    return prompt_file.read_text(encoding="utf-8").strip()


def _ensure_client(
    openai_client: Optional[OpenAI] = None, api_key: Optional[str] = None
) -> OpenAI:
    """Return an OpenAI client, lazily creating one from various sources.

    This function implements a singleton pattern for OpenAI client creation, using
    cached instances to avoid repeated initialization. If no client is provided,
    it creates one using the provided api_key, docassemble config, or environment variables
    OPENAI_API_KEY and optionally OPENAI_ORGANIZATION/OPENAI_ORG

    Args:
        openai_client: Pre-configured OpenAI client. If provided, this client
            is returned directly without any caching or initialization.
        api_key: Explicit API key to use for client creation.

    Returns:
        An initialized OpenAI client ready for API calls.

    Raises:
        RuntimeError: If no API key can be found from any source and
            no client is provided.
    """

    if openai_client:
        return openai_client

    global _cached_client, _api_key, _organization

    # If an explicit API key is provided, create a new client (don't cache)
    if api_key:
        return OpenAI(api_key=api_key, organization=_organization or None)

    if _cached_client is None:
        resolved_key = get_openai_api_key()
        if not resolved_key:
            raise RuntimeError(
                "OPENAI_API_KEY must be set (environment variable, docassemble config, or passed explicitly) to use passive voice detection."
            )
        _cached_client = OpenAI(
            api_key=resolved_key, organization=_organization or None
        )

    return _cached_client


_SENTENCE_REGEX = re.compile(
    r"[^.!?\n]+(?:[.!?](?!['\"]?\w))?",
    re.UNICODE,
)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern matching.

    It uses a regex to identify sentence boundaries
    based on periods, exclamation marks, and question marks, while avoiding splits
    on abbreviations followed by quoted words.

    Args:
        text: Input text to split into sentences.

    Returns:
        List of sentence strings with leading/trailing whitespace removed.
        Empty strings are filtered out from the results.

    Note:
        It replaces the original NLTK-based splitter to reduce dependency weight and
        package conflicts.
    """

    candidates = [segment.strip() for segment in _SENTENCE_REGEX.findall(text)]
    return [segment for segment in candidates if segment]


def _normalize_input(text: Union[str, Sequence[str]]) -> List[str]:
    """Convert input text into a normalized list of sentences for analysis.

    This function handles both string and sequence inputs, performing sentence
    splitting for strings and validation/filtering for sequences. Only sentences
    with more than 2 words are retained, as shorter fragments are typically
    not meaningful for passive voice analysis.

    Args:
        text: Input text as either a single string (which will be split into
            sentences) or a sequence of strings (which will be validated
            and filtered).

    Returns:
        List of sentence strings, each containing more than 2 words with
        leading/trailing whitespace removed.

    Raises:
        ValueError: If input is not a string or sequence of strings, if any
            item in a sequence is not a string, or if no valid sentences
            (>2 words) are found in the input.
    """

    if isinstance(text, str):
        sentences = [s for s in split_sentences(text) if len(s.split()) > 2]
    elif isinstance(text, Sequence):
        sentences = []
        for item in text:
            if not isinstance(item, str):
                raise ValueError(
                    "Passive voice detector only accepts strings or sequences of strings."
                )
            cleaned = item.strip()
            if len(cleaned.split()) > 2:
                sentences.append(cleaned)
    else:
        raise ValueError(
            "Passive voice detector only accepts a string or a sequence of strings."
        )

    if not sentences:
        raise ValueError(
            "There are no sentences over 2 words in the provided text to analyze."
        )
    return sentences


def _extract_text_from_response(response) -> str:
    """Extract text content from OpenAI ChatCompletion API response object.

    See https://platform.openai.com/docs/api-reference/responses/object?lang=python for more information about the OpenAI's possible return types.

    Args:
        response: OpenAI API response object with potentially nested content
            structures containing text data.

    Returns:
        Concatenated text content from the response, with leading/trailing
        whitespace removed. Returns empty string if no text content is found.
    """

    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    chunks: List[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)

    if not chunks and hasattr(response, "data"):
        for item in response.data:
            text = getattr(item, "text", None)
            if text:
                chunks.append(text)

    return "".join(chunks).strip()


def detect_passive_voice_segments(
    text: Union[str, Sequence[str]],
    openai_client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> List[Tuple[str, List[str]]]:
    """Detect passive voice constructions in text using OpenAI's language model.

    This function analyzes sentences to identify passive voice constructions by
    leveraging OpenAI's Chat Completions API with a simple single-prompt approach.

    Args:
        text: Input text as either a single string (which will be split into
            sentences) or a sequence of pre-split sentences to analyze.
        openai_client: Pre-configured OpenAI client instance. If None, a client
            will be created using the api_key parameter or environment variables.
        model: OpenAI model identifier to use for analysis. Defaults to the
            value of DEFAULT_MODEL constant (currently 'gpt-5-nano').
        api_key: Explicit OpenAI API key to use. If None, will try docassemble
            config (if available) then environment variables (OPENAI_API_KEY required).

    Returns:
        List of tuples where each tuple contains:
        - sentence (str): The original sentence text
        - fragments (List[str]): List of passive voice text fragments found
          in the sentence, or empty list if no passive voice detected

    Raises:
        ValueError: If input text contains no valid sentences (>2 words) or
            if input format is invalid.
        RuntimeError: If no API key can be found from any source and no client provided.
        AuthenticationError: If OpenAI API authentication fails (may retry
            once without organization header).

    Example:
        >>> detect_passive_voice_segments("The ball was thrown by John.")
        [('The ball was thrown by John.', ['was thrown'])]

        >>> detect_passive_voice_segments("John threw the ball.")
        [('John threw the ball.', [])]

    Note:
        This implementation uses a single prompt per sentence to classify rather
        than the ChatCompletion API after testing and finding better performance with this
        simple approach.
    """

    sentences = _normalize_input(text)
    client = _ensure_client(openai_client, api_key)

    ordered_results = []

    for sentence in sentences:
        system_prompt = _load_prompt()
        system_message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}],
        }
        user_message: ChatCompletionUserMessageParam = {
            "role": "user",
            "content": [{"type": "text", "text": f"Sentence: {sentence}"}],
        }
        messages: List[ChatCompletionMessageParam] = [system_message, user_message]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=500,  # We only need one word: "passive" or "active", but leave room for reasoning tokens with gpt-5
            )
        except AuthenticationError as exc:
            global _organization, _cached_client, _api_key
            if "mismatched_organization" in str(exc).lower() and _organization:
                # Retry without the organization header.
                _organization = None
                retry_key = api_key or get_openai_api_key() or _api_key
                new_client = OpenAI(api_key=retry_key)
                _cached_client = new_client
                response = new_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=500,
                )
            else:
                raise

        content = response.choices[0].message.content
        if content:
            classification = content.strip().lower()

            if classification == "passive":
                # If classified as passive, return the whole sentence as the fragment, to match return type of passivepy
                # Note that when we had fragment detection in the prompt, performance dropped significantly
                fragments = [sentence]
            else:
                # If active or any other response, return empty list
                fragments = []
        else:
            # No response content, assume active
            fragments = []

        ordered_results.append((sentence, fragments))

    return ordered_results
