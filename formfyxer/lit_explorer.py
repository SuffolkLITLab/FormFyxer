import os
import re
import subprocess
import tempfile

from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

import pikepdf
import textstat
import requests
import json
import numpy as np
import pandas as pd
from numpy import unique
from numpy import where

import eyecite
from enum import Enum
import sigfig
import yaml
from .pdf_wrangling import (
    get_existing_pdf_fields,
    FormField,
    FieldType,
    unlock_pdf_in_place,
    is_tagged,
    get_original_text_with_fields,
)

import math
from contextlib import contextmanager
from functools import lru_cache
import threading
import _thread
from typing import (
    Any,
    Optional,
    Union,
    Iterable,
    List,
    Dict,
    Tuple,
    Callable,
    TypedDict,
    cast,
)

import openai
from openai import OpenAI
from dotenv import load_dotenv
import tiktoken

from .passive_voice_detection import detect_passive_voice_segments, split_sentences
from .docassemble_support import get_openai_api_key_from_sources

from pathlib import Path

load_dotenv()

class OpenAiCreds(TypedDict):
    org: str
    key: str


def _load_prompt(prompt_name: str) -> str:
    """Load a prompt from the prompts directory.
    
    Args:
        prompt_name: Name of the prompt file (without .txt extension)
        
    Returns:
        The prompt text as a string.
        
    Raises:
        FileNotFoundError: If the prompt file cannot be found.
    """
    current_dir = Path(__file__).parent
    prompt_file = current_dir / "prompts" / f"{prompt_name}.txt"
    return prompt_file.read_text(encoding="utf-8").strip()


DEFAULT_TIKTOKEN_ENCODING = "cl100k_base"


@lru_cache(maxsize=16)
def _resolve_encoding(model_name: Optional[str] = None):
    """Resolve a tiktoken encoder for the provided model name with caching."""
    try:
        if model_name:
            return tiktoken.encoding_for_model(model_name)
    except (KeyError, ValueError):
        pass
    return tiktoken.get_encoding(DEFAULT_TIKTOKEN_ENCODING)


def _token_count(text: Optional[str], model_name: Optional[str] = None) -> int:
    """Return the number of tokens in ``text`` for the given model."""
    if not text:
        return 0
    encoding = _resolve_encoding(model_name)
    return len(encoding.encode(text))


def _truncate_to_token_limit(
    text: str, max_tokens: int, model_name: Optional[str] = None
) -> str:
    """Truncate ``text`` to ``max_tokens`` for the provided model using tiktoken."""
    if max_tokens <= 0 or not text:
        return "" if max_tokens <= 0 else text

    encoding = _resolve_encoding(model_name)
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens])

STOP_WORDS = {
    'a','about','above','after','again','against','all','am','an','and','any','are','aren','as','at',
    'be','because','been','before','being','below','between','both','but','by',
    'could','did','do','does','doing','down','during',
    'each','few','for','from','further',
    'had','has','have','having','he','her','here','hers','herself','him','himself','his','how',
    'i','if','in','into','is','it','its','itself',
    'just',
    'me','more','most','my','myself',
    'no','nor','not',
    'of','off','on','once','only','or','other','our','ours','ourselves','out','over','own',
    'same','she','should','so','some','such',
    'than','that','the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too',
    'under','until','up','very',
    'was','we','were','what','when','where','which','while','who','whom','why','will','with','you','your','yours','yourself','yourselves'
}

# Load local variables, models, and API key(s).

default_spot_token = os.getenv("SPOT_TOKEN") or os.getenv("TOOLS_TOKEN")
default_key: Optional[str] = os.getenv("OPENAI_API_KEY")
default_org: Optional[str] = (
    os.getenv("OPENAI_ORGANIZATION")
    or os.getenv("OPENAI_ORG")
    or os.getenv("OPENAI_ORG_ID")
)
if default_key:
    client: Optional[OpenAI] = OpenAI(
        api_key=default_key, organization=default_org or None
    )
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI()
else:
    client = None


# TODO(brycew): remove by retraining the model to work with random_state=4.
NEEDS_STABILITY = True if os.getenv("ISUNITTEST") else False

# Define some hardcoded data file paths

CURRENT_DIRECTORY = os.path.dirname(__file__)
GENDERED_TERMS_PATH = os.path.join(CURRENT_DIRECTORY, "data", "gendered_terms.yml")
PLAIN_LANGUAGE_TERMS_PATH = os.path.join(
    CURRENT_DIRECTORY, "data", "simplified_words.yml"
)


# This creates a timeout exception that can be triggered when something hangs too long.
class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out.")
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def recursive_get_id(values_to_unpack: Union[dict, list], tmpl: Optional[set] = None):
    """
    Pull ID values out of the LIST/NSMI results from Spot.
    """
    # h/t to Quinten and Bryce for this code ;)
    if not tmpl:
        tmpl = set()
    if isinstance(values_to_unpack, dict):
        tmpl.add(values_to_unpack.get("id"))
        if values_to_unpack.get("children"):
            tmpl.update(recursive_get_id(values_to_unpack.get("children", []), tmpl))
        return tmpl
    elif isinstance(values_to_unpack, list):
        for item in values_to_unpack:
            tmpl.update(recursive_get_id(item, tmpl))
        return tmpl
    else:
        return set()


def spot(
    text: str,
    lower: float = 0.25,
    pred: float = 0.5,
    upper: float = 0.6,
    verbose: float = 0,
    token: str = "",
):
    """
    Call the Spot API (https://spot.suffolklitlab.org) to classify the text of a PDF using
    the NSMIv2/LIST taxonomy (https://taxonomy.legal/), but returns only the IDs of issues found in the text.
    """
    global default_spot_token
    if not token:
        if not default_spot_token:
            print("You need to pass a spot token when using Spot")
            return []
        token = default_spot_token
    headers = {
        "Authorization": "Bearer " + token,
        "Content-Type": "application/json",
    }

    body = {
        "text": text[:5000],
        "save-text": 0,
        "cutoff-lower": lower,
        "cutoff-pred": pred,
        "cutoff-upper": upper,
    }
    r = requests.post(
        "https://spot.suffolklitlab.org/v0/entities-nested/",
        headers=headers,
        data=json.dumps(body),
    )
    output_ = r.json()
    try:
        output_["build"]
        if verbose != 1:
            try:
                return list(recursive_get_id(output_["labels"]))
            except:
                return []
        else:
            return output_
    except:
        return output_


# A function to pull words out of snake_case, camelCase and the like.


def re_case(text: str) -> str:
    """
    Capture PascalCase, snake_case and kebab-case terms and add spaces to separate the joined words
    """
    re_outer = re.compile(r"([^A-Z ])([A-Z])")
    re_inner = re.compile(r"(?<!^)([A-Z])([^A-Z])")
    text = re_outer.sub(r"\1 \2", re_inner.sub(r" \1\2", text))
    return text.replace("_", " ").replace("-", " ")


# Takes text from an auto-generated field name and uses regex to convert it into an Assembly Line standard field.
# See https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/label_variables/


def regex_norm_field(text: str):
    """
    Apply some heuristics to a field name to see if we can get it to match AssemblyLine conventions.
    See: https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/document_variables
    """
    regex_list = [
        # Personal info
        ## Name & Bio
        [r"^((My|Your|Full( legal)?) )?Name$", "users1_name"],
        [r"^(Typed or )?Printed Name\s?\d*$", "users1_name"],
        [r"^(DOB|Date of Birth|Birthday)$", "users1_birthdate"],
        ## Address
        [r"^(Street )?Address$", "users1_address_line_one"],
        [r"^City State Zip$", "users1_address_line_two"],
        [r"^City$", "users1_address_city"],
        [r"^State$", "users1_address_state"],
        [r"^Zip( Code)?$", "users1_address_zip"],
        ## Contact
        [r"^(Phone|Telephone)$", "users1_phone_number"],
        [r"^Email( Address)$", "users1_email"],
        # Parties
        [r"^plaintiff\(?s?\)?$", "plaintiff1_name"],
        [r"^defendant\(?s?\)?$", "defendant1_name"],
        [r"^petitioner\(?s?\)?$", "petitioners1_name"],
        [r"^respondent\(?s?\)?$", "respondents1_name"],
        # Court info
        [r"^(Court\s)?Case\s?(No|Number)?\s?A?$", "docket_number"],
        [r"^file\s?(No|Number)?\s?A?$", "docket_number"],
        # Form info
        [r"^(Signature|Sign( here)?)\s?\d*$", "users1_signature"],
        [r"^Date\s?\d*$", "signature_date"],
    ]
    for regex in regex_list:
        text = re.sub(regex[0], regex[1], text, flags=re.IGNORECASE)
    return text


def reformat_field(text: str, max_length: int = 30, tools_token: Optional[str] = None):
    """Generate a snake_case label from ``text`` without external similarity scoring."""
    orig_title = text.lower()
    orig_title = re.sub(r"[^a-zA-Z]+", " ", orig_title)
    orig_title_words = orig_title.split()
    deduped_sentence = []
    for word in orig_title_words:
        if word not in deduped_sentence:
            deduped_sentence.append(word)
    # Use a local hardcoded stop word list (exported from passive voice detection)
    filtered_sentence = [w for w in deduped_sentence if w.lower() not in STOP_WORDS]
    candidate_words = filtered_sentence or deduped_sentence

    sanitized_words: List[str] = []
    for word in candidate_words:
        cleaned = re.sub(r"[^a-z0-9]", "", word.lower())
        if cleaned:
            sanitized_words.append(cleaned)

    if not sanitized_words:
        sanitized_words = [
            re.sub(r"[^a-z0-9]", "", word.lower()) for word in orig_title_words
        ]
        sanitized_words = [word for word in sanitized_words if word]

    if sanitized_words:
        new_words: List[str] = []
        remaining_length = max_length if max_length > 0 else 0

        for word in sanitized_words:
            if remaining_length <= 0:
                break

            if not new_words:
                trimmed_word = word[:remaining_length] if remaining_length else ""
                if trimmed_word:
                    new_words.append(trimmed_word)
                    remaining_length -= len(trimmed_word)
            else:
                if remaining_length <= 1:
                    break  # not enough space for separator + word
                remaining_length -= 1  # account for underscore
                if remaining_length <= 0:
                    break
                trimmed_word = word[:remaining_length]
                if trimmed_word:
                    new_words.append(trimmed_word)
                    remaining_length -= len(trimmed_word)
                else:
                    break

        if new_words:
            return "_".join(new_words)

    if re.search(r"^(\d+)$", text):
        return "unknown"
    return re.sub(r"\s+", "_", text.lower())


def normalize_name(
    jur: str,
    group: str,
    n: int,
    per,
    last_field: str,
    this_field: str,
    tools_token: Optional[str] = None,
    context: Optional[str] = None,
    openai_creds: Optional[OpenAiCreds] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5-nano",
) -> Tuple[str, float]:
    """
    Normalize a field name, if possible to the Assembly Line conventions, and if
    not, to a snake_case variable name of appropriate length.

    In most cases, you should use the better performing `rename_pdf_fields_with_context` function,
    which renames all fields in one prompt to an LLM.

    Args:
        jur: Jurisdiction (legacy parameter, maintained for compatibility)
        group: Group/category (legacy parameter, maintained for compatibility)  
        n: Position in field list (legacy parameter, maintained for compatibility)
        per: Percentage through field list (legacy parameter, maintained for compatibility)
        last_field: Previous field name (legacy parameter, maintained for compatibility)
        this_field: The field name to normalize
        tools_token: Tools API token (legacy parameter, maintained for compatibility)
        context: Optional PDF text context to help with field naming
        openai_creds: OpenAI credentials for LLM calls
        api_key: OpenAI API key (overrides creds and env vars)
        model: OpenAI model to use (default: gpt-5-nano)
    
    Returns:
        Tuple of (normalized_field_name, confidence_score)
    
    If context and LLM credentials are provided, uses LLM normalization.
    Otherwise, falls back to traditional regex-based approach for backward compatibility.
    """
    
    # Note: previous versions relied on a hardcoded `included_fields` list
    # to short-circuit normalization for known Assembly Line fields. That list
    # has been removed. We now always attempt LLM-assisted normalization when
    # context and credentials are available, falling back to the traditional
    # regex-based and reformat approach below.
    
    # If context and LLM credentials are provided, use enhanced LLM normalization
    if context and (openai_creds or api_key or os.getenv("OPENAI_API_KEY")):
        try:
            # Use LLM to normalize the field name with context
            system_message = _load_prompt("normalize_field_name")
            
            # Truncate context if too long (keep reasonable size for token limits)
            max_context_chars = 2000  # Roughly 500 tokens
            truncated_context = context[:max_context_chars] if len(context) > max_context_chars else context
            
            user_message = f"""Field to normalize: "{this_field}"

Context from PDF form:
{truncated_context}

Additional information:
- Field position: {n} out of total fields
- Previous field: "{last_field}"
- Jurisdiction: "{jur}" (if relevant)
- Category: "{group}" (if relevant)

Please normalize this field name following Assembly Line conventions."""

            # Resolve API key
            resolved_api_key = api_key or (openai_creds["key"] if openai_creds else None) or os.getenv("OPENAI_API_KEY")
            
            response = text_complete(
                system_message=system_message,
                user_message=user_message,
                max_tokens=500,  # Small response expected
                creds=openai_creds,
                api_key=resolved_api_key,
                model=model,
            )
            
            # Parse the response
            if isinstance(response, dict):
                normalized_name = response.get("normalized_name", "")
                confidence = response.get("confidence", 0.5)
            elif isinstance(response, str):
                try:
                    parsed_response = json.loads(response)
                    normalized_name = parsed_response.get("normalized_name", "")
                    confidence = parsed_response.get("confidence", 0.5)
                except json.JSONDecodeError:
                    raise ValueError(f"Failed to parse JSON response: {response}")
            else:
                raise ValueError(f"Unexpected response type: {type(response)}")
            
            # Validate the response
            if not normalized_name or not isinstance(normalized_name, str):
                raise ValueError("No valid normalized_name in response")
            
            # Basic validation: ensure it follows snake_case conventions
            if not re.match(r'^[a-z][a-z0-9_]*$', normalized_name):
                # If the LLM response doesn't follow conventions, clean it up
                normalized_name = re.sub(r'[^a-z0-9_]', '_', normalized_name.lower())
                normalized_name = re.sub(r'_+', '_', normalized_name)  # Remove multiple underscores
                normalized_name = normalized_name.strip('_')  # Remove leading/trailing underscores
                if not normalized_name or not normalized_name[0].isalpha():
                    # Fallback if still invalid
                    raise ValueError("Invalid field name after cleanup")
                confidence = max(0.1, confidence - 0.2)  # Reduce confidence for cleaned names
            
            # Ensure confidence is in valid range
            confidence = max(0.1, min(1.0, float(confidence)))
            
            return normalized_name, confidence
            
        except Exception as ex:
            print(f"LLM field normalization failed for '{this_field}': {ex}, falling back to traditional approach")
            # Fall through to traditional approach below
    
    # Traditional approach (original behavior)
    # Re-case and normalize the field using regex rules
    processed_field = re_case(this_field)
    processed_field = regex_norm_field(processed_field)
    
    return reformat_field(processed_field, tools_token=tools_token), 0.5


def rename_pdf_fields_with_context(
    pdf_path: str,
    original_field_names: List[str],
    openai_creds: Optional[OpenAiCreds] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5-nano",
) -> Dict[str, str]:
    """
    Use LLM to rename PDF fields based on full PDF context with field markers.
    
    Args:
        pdf_path: Path to the PDF file
        original_field_names: List of original field names from the PDF
        openai_creds: OpenAI credentials to use for the API call
        api_key: explicit API key to use (overrides creds and env vars)
        model: the OpenAI model to use (default: gpt-5-nano)
    
    Returns:
        Dictionary mapping original field names to new Assembly Line names
    """
    if not original_field_names:
        return {}
    
    try:
        # Get PDF text with field markers
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
            try:
                get_original_text_with_fields(pdf_path, temp_file.name)
                
                # Read the text with field markers
                with open(temp_file.name, 'r', encoding='utf-8') as f:
                    pdf_text_with_fields = f.read()
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file.name)
                except:
                    pass
        
        if not pdf_text_with_fields or not pdf_text_with_fields.strip():
            # Fallback: if we can't get text with field markers, use basic approach
            print("Warning: Could not extract PDF text with field markers, falling back to regex approach")
            return {name: regex_norm_field(re_case(name)) for name in original_field_names}
        
        # Load the field labeling prompt
        system_message = _load_prompt("field_labeling")
        
        # For GPT-5-nano: Support up to 30 pages (roughly 100K tokens input, well within 400K limit)
        # Estimate: 30 pages * ~1300 tokens/page = ~39K tokens for PDF text
        # Plus prompt and field list = ~50K total input tokens (comfortable margin)
        max_pdf_text_chars = 300000  # Roughly 75K tokens worth of text
        
        user_message = f"""Here is the PDF form text with field markers:

{pdf_text_with_fields[:max_pdf_text_chars]}

Original field names to rename:
{json.dumps(original_field_names, indent=2)}

Please analyze the context around each field marker and provide appropriate Assembly Line variable names."""

        # Call the LLM with much higher limits for GPT-5-nano
        response = text_complete(
            system_message=system_message,
            user_message=user_message,
            max_tokens=15000,  # Increased for larger field lists and more detailed reasoning
            creds=openai_creds,
            api_key=api_key,
            model=model,
        )
        
        # Parse the response
        if isinstance(response, dict):
            field_mappings = response.get("field_mappings", {})
        elif isinstance(response, str):
            try:
                parsed_response = json.loads(response)
                field_mappings = parsed_response.get("field_mappings", {})
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON response: {response}")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        # Validate the response
        if not isinstance(field_mappings, dict):
            raise ValueError("field_mappings is not a dictionary")
        
        # Ensure all original fields are mapped
        mapped_fields = set(field_mappings.keys())
        original_fields = set(original_field_names)
        
        missing_fields = original_fields - mapped_fields
        extra_fields = mapped_fields - original_fields
        
        # Handle missing fields with fallback
        for missing_field in missing_fields:
            fallback_name = regex_norm_field(re_case(missing_field))
            field_mappings[missing_field] = fallback_name
            print(f"Warning: LLM didn't map '{missing_field}', using fallback: '{fallback_name}'")
        
        # Remove extra fields that weren't in the original list
        for extra_field in extra_fields:
            del field_mappings[extra_field]
            print(f"Warning: LLM provided mapping for unknown field '{extra_field}', removing")
        
        # Handle duplicates by adding suffixes
        final_mappings = {}
        used_names = set()
        
        for original_name in original_field_names:
            new_name = field_mappings.get(original_name, original_name)
            
            # If this name is already used, add a suffix
            if new_name in used_names:
                counter = 2
                base_name = new_name
                while f"{base_name}__{counter}" in used_names:
                    counter += 1
                new_name = f"{base_name}__{counter}"
            
            final_mappings[original_name] = new_name
            used_names.add(new_name)
        
        return final_mappings
        
    except Exception as ex:
        print(f"Failed to rename fields with LLM: {ex}")
        
        # Fallback: use regex-based approach
        fallback_mappings = {}
        used_names = set()
        
        for original_name in original_field_names:
            new_name = regex_norm_field(re_case(original_name))
            
            # Handle duplicates
            if new_name in used_names:
                counter = 2
                base_name = new_name
                while f"{base_name}__{counter}" in used_names:
                    counter += 1
                new_name = f"{base_name}__{counter}"
            
            fallback_mappings[original_name] = new_name
            used_names.add(new_name)
        
        return fallback_mappings


# Take a list of AL variables and spits out suggested groupings. Here's what's going on:
#
# 1. It reads in a list of fields (e.g., `["user_name","user_address"]`)
# 2. Splits each field into words (e.g., turning `user_name` into `user name`)
# 3. It then turns these ngrams/"sentences" into vectors using word2vec.
# 4. For the collection of fields, it finds clusters of these "sentences" within the semantic space defined by word2vec. Currently it uses Affinity Propagation. See https://machinelearningmastery.com/clustering-algorithms-with-python/


def cluster_screens(
    fields: List[str] = [],
    openai_creds: Optional[OpenAiCreds] = None,
    api_key: Optional[str] = None,
    model: str = "gpt-5-nano",
    damping: Optional[float] = None,
    tools_token: Optional[str] = None,
) -> Dict[str, List[str]]:
    """
    Groups the given fields into screens using an LLM (GPT) for semantic understanding.

    Args:
      fields: a list of field names
      openai_creds: OpenAI credentials to use for the API call
      api_key: explicit API key to use (overrides creds and env vars)
      model: the OpenAI model to use (default: gpt-5-nano, can use gpt-4 variants)
      damping: deprecated parameter, kept for backward compatibility
      tools_token: deprecated parameter, kept for backward compatibility

    Returns: a suggested screen grouping, each screen name mapped to the list of fields on it
    """
    if not fields:
        return {}

    # Create system and user messages for the LLM to group fields logically
    system_message = _load_prompt("field_grouping")

    user_message = f"""Here are the field names to group:
{json.dumps(fields, indent=2)}

Please group these fields into logical screens following the guidelines provided."""

    response: Union[str, Dict[str, Any]] = ""
    try:
        # Use the text_complete function to call the LLM
        response = text_complete(
            system_message=system_message,
            user_message=user_message,
            max_tokens=3000,
            creds=openai_creds,
            api_key=api_key,
            model=model,
        )

        # Handle the response (could be dict if JSON was parsed, or str if parsing failed)
        if isinstance(response, dict):
            screens = cast(Dict[str, List[str]], response)
        elif isinstance(response, str):
            # If we got a string back, the JSON parsing failed - try manual parsing as fallback
            try:
                screens = json.loads(response)
            except json.JSONDecodeError:
                raise ValueError(f"Failed to parse JSON response: {response}")
        else:
            raise ValueError(f"Unexpected response type: {type(response)}")
        
        # Validate the response
        if not isinstance(screens, dict):
            raise ValueError("Response is not a dictionary")
        
        # Collect all fields from the response
        response_fields = []
        for screen_fields in screens.values():
            if not isinstance(screen_fields, list):
                raise ValueError(f"Screen fields must be a list, got {type(screen_fields)}")
            response_fields.extend(screen_fields)
        
        # Check that all input fields are present in the output
        input_set = set(fields)
        response_set = set(response_fields)
        
        missing_fields = input_set - response_set
        extra_fields = response_set - input_set
        
        if missing_fields or extra_fields:
            raise ValueError(
                f"Field validation failed. Missing: {missing_fields}, Extra: {extra_fields}"
            )
        
        # Check for duplicate fields in the response
        if len(response_fields) != len(response_set):
            field_counts: Dict[str, int] = {}
            for field in response_fields:
                field_counts[field] = field_counts.get(field, 0) + 1
            duplicates = [field for field, count in field_counts.items() if count > 1]
            raise ValueError(f"Duplicate fields found in response: {duplicates}")
        
        return screens
        
    except Exception as ex:
        print(f"Failed to parse LLM response or validation failed: {ex}")
        if hasattr(ex, '__context__') and ex.__context__:
            print(f"Context: {ex.__context__}")
        print(f"Response: {response}")
        
        # Fallback: create a simple grouping based on field name patterns
        return _fallback_field_grouping(fields)


def _fallback_field_grouping(fields: List[str]) -> Dict[str, List[str]]:
    """
    Fallback field grouping when LLM fails. Groups fields based on simple heuristics.
    """
    if not fields:
        return {}
    
    screens = {}
    personal_info = []
    party_info = []
    case_info = []
    signature_info = []
    other_fields = []
    
    # Simple keyword-based grouping
    for field in fields:
        field_lower = field.lower()
        if any(keyword in field_lower for keyword in ['name', 'address', 'phone', 'email', 'birth']):
            personal_info.append(field)
        elif any(keyword in field_lower for keyword in ['plaintiff', 'defendant', 'petitioner', 'respondent']):
            party_info.append(field)
        elif any(keyword in field_lower for keyword in ['docket', 'case', 'court', 'trial']):
            case_info.append(field)
        elif any(keyword in field_lower for keyword in ['signature', 'date']):
            signature_info.append(field)
        else:
            other_fields.append(field)
    
    # Only add non-empty screens
    if personal_info:
        screens['personal_information'] = personal_info
    if party_info:
        screens['party_information'] = party_info
    if case_info:
        screens['case_information'] = case_info
    if signature_info:
        screens['signatures_and_dates'] = signature_info
    if other_fields:
        screens['other_fields'] = other_fields
    
    # If no fields were categorized, put them all in one screen
    if not screens:
        screens['screen_1'] = fields
    
    return screens


def get_character_count(
    field: FormField, char_width: float = 6, row_height: float = 16
) -> int:
    # https://pikepdf.readthedocs.io/en/latest/api/main.html#pikepdf.Rectangle
    # Rectangle with llx,lly,urx,ury
    height = field.configs.get("height") or field.configs.get("size", 0)
    width = field.configs.get("width") or field.configs.get("size", 0)
    num_rows = int(height / row_height) if height > row_height else 1  # type: ignore
    num_cols = int(width / char_width)  # type: ignore
    max_chars = num_rows * num_cols
    return min(max_chars, 1)


class InputType(Enum):
    """
    Input type maps onto the type of input the PDF author chose for the field. We only
    handle text, checkbox, and signature fields.
    """

    TEXT = "Text"
    CHECKBOX = "Checkbox"
    SIGNATURE = "Signature"

    def __str__(self):
        return self.value


class FieldInfo(TypedDict):
    var_name: str
    max_length: int
    type: Union[InputType, str]


def field_types_and_sizes(
    fields: Optional[Iterable[FormField]],
) -> List[FieldInfo]:
    """
    Transform the fields provided by get_existing_pdf_fields into a summary format.
    Result will look like:
    [
        {
            "var_name": var_name,
            "type": "text | checkbox | signature",
            "max_length": n
        }
    ]
    """
    processed_fields: List[FieldInfo] = []
    if not fields:
        return []
    for field in fields:
        item: FieldInfo = {
            "var_name": field.name,
            "max_length": get_character_count(
                field,
            ),
            "type": "",
        }
        if field.type == FieldType.TEXT or field.type == FieldType.AREA:
            item["type"] = InputType.TEXT
        elif field.type == FieldType.CHECK_BOX:
            item["type"] = InputType.CHECKBOX
        elif field.type == FieldType.SIGNATURE:
            item["type"] = InputType.SIGNATURE
        else:
            item["type"] = str(field.type)
        processed_fields.append(item)
    return processed_fields


class AnswerType(Enum):
    """
    Answer type describes the effort the user answering the form will require.
    "Slot-in" answers are a matter of almost instantaneous recall, e.g., name, address, etc.
    "Gathered" answers require looking around one's desk, for e.g., a health insurance number.
    "Third party" answers require picking up the phone to call someone else who is the keeper
    of the information.
    "Created" answers don't exist before the user is presented with the question. They may include
    a choice, creating a narrative, or even applying legal reasoning. "Affidavits" are a special
    form of created answers.
    See Jarret and Gaffney, Forms That Work (2008)
    """

    SLOT_IN = "Slot in"
    GATHERED = "Gathered"
    THIRD_PARTY = "Third party"
    CREATED = "Created"
    AFFIDAVIT = "Affidavit"

    def __str__(self):
        return self.value


def classify_field(field: FieldInfo, new_name: str) -> AnswerType:
    """
    Apply heuristics to the field's original and "normalized" name to classify
    it as either a "slot-in", "gathered", "third party" or "created" field type.
    """
    SLOT_IN_FIELDS = {
        "users1_name",
        "users1_name",
        "users1_birthdate",
        "users1_address_line_one",
        "users1_address_line_two",
        "users1_address_city",
        "users1_address_state",
        "users1_address_zip",
        "users1_phone_number",
        "users1_email",
        "plaintiff1_name",
        "defendant1_name",
        "petitioners1_name",
        "respondents1_name",
        "users1_signature",
        "signature_date",
    }
    SLOT_IN_KEYWORDS = {
        "name",
        "birth date",
        "birthdate",
        "phone",
    }
    GATHERED_KEYWORDS = {
        "number",
        "value",
        "amount",
        "id number",
        "social security",
        "benefit id",
        "docket",
        "case",
        "employer",
        "date",
    }
    CREATED_KEYWORDS = {
        "choose",
        "choice",
        "why",
        "fact",
    }
    AFFIDAVIT_KEYWORDS = {
        "affidavit",
    }
    var_name = field["var_name"].lower()
    if (
        var_name in SLOT_IN_FIELDS
        or new_name in SLOT_IN_FIELDS
        or any(keyword in var_name for keyword in SLOT_IN_KEYWORDS)
    ):
        return AnswerType.SLOT_IN
    elif any(keyword in var_name for keyword in GATHERED_KEYWORDS):
        return AnswerType.GATHERED
    elif set(var_name.split()).intersection(CREATED_KEYWORDS):
        return AnswerType.CREATED
    elif field["type"] == InputType.TEXT:
        if field["max_length"] <= 100:
            return AnswerType.SLOT_IN
        else:
            return AnswerType.CREATED
    return AnswerType.GATHERED


def get_adjusted_character_count(field: FieldInfo) -> float:
    """
    Determines the bracketed length of an input field based on its max_length attribute,
    returning a float representing the approximate length of the field content.

    The function chunks the answers into 5 different lengths (checkboxes, 2 words, short, medium, and long)
    instead of directly using the character count, as forms can allocate different spaces
    for the same data without considering the space the user actually needs.

    Args:
        field (FieldInfo): An object containing information about the input field,
                           including the "max_length" attribute.

    Returns:
        float: The approximate length of the field content, categorized into checkboxes, 2 words, short,
               medium, or long based on the max_length attribute.

    Examples:
        >>> get_adjusted_character_count({"type"}: InputType.CHECKBOX)
        4.7
        >>> get_adjusted_character_count({"max_length": 100})
        9.4
        >>> get_adjusted_character_count({"max_length": 300})
        230
        >>> get_adjusted_character_count({"max_length": 600})
        115
        >>> get_adjusted_character_count({"max_length": 1200})
        1150
    """
    ONE_WORD = 4.7  # average word length: https://www.researchgate.net/figure/Average-word-length-in-the-English-language-Different-colours-indicate-the-results-for_fig1_230764201
    ONE_LINE = 115  # Standard line is ~ 115 characters wide at 12 point font
    SHORT_ANSWER = (
        ONE_LINE * 2
    )  # Anything over 1 line but less than 3 probably needs about the same time to answer
    MEDIUM_ANSWER = ONE_LINE * 5
    LONG_ANSWER = (
        ONE_LINE * 10
    )  # Anything over 10 lines probably needs a full page but form author skimped on space
    if field["type"] != InputType.TEXT:
        return ONE_WORD

    if field["max_length"] <= ONE_LINE or (field["max_length"] <= ONE_LINE * 2):
        return ONE_WORD * 2
    elif field["max_length"] <= SHORT_ANSWER:
        return SHORT_ANSWER
    elif field["max_length"] <= MEDIUM_ANSWER:
        return MEDIUM_ANSWER
    return LONG_ANSWER


def time_to_answer_field(
    field: FieldInfo,
    new_name: str,
    cpm: int = 40,
    cpm_std_dev: int = 17,
) -> Callable[[int], np.ndarray]:
    """
    Apply a heuristic for the time it takes to answer the given field, in minutes.
    It is hand-written for now.
    It will factor in the input type, the answer type (slot in, gathered, third party or created), and the
    amount of input text allowed in the field.
    The return value is a function that can return N samples of how long it will take to answer the field (in minutes)
    """
    # Average CPM is about 40: https://en.wikipedia.org/wiki/Words_per_minute#Handwriting
    # Standard deviation is about 17 characters/minute
    # Add mean amount of time for gathering or creating the answer itself (if any) + standard deviation in minutes
    TIME_TO_MAKE_ANSWER = {
        AnswerType.SLOT_IN: (0.25, 0.1),
        AnswerType.GATHERED: (3, 2),
        AnswerType.THIRD_PARTY: (5, 2),
        AnswerType.CREATED: (5, 4),
        AnswerType.AFFIDAVIT: (5, 4),
    }
    kind = classify_field(field, new_name)
    if field["type"] == InputType.SIGNATURE or "signature" in field["var_name"]:
        return lambda number_samples: np.random.normal(
            loc=0.5, scale=0.1, size=number_samples
        )
    if field["type"] == InputType.CHECKBOX:
        return lambda number_samples: np.random.normal(
            loc=TIME_TO_MAKE_ANSWER[kind][0],
            scale=TIME_TO_MAKE_ANSWER[kind][1],
            size=number_samples,
        )
    else:
        adjusted_character_count = get_adjusted_character_count(field)
        time_to_write_answer = adjusted_character_count / cpm
        time_to_write_std_dev = adjusted_character_count / cpm_std_dev

        return lambda number_samples: np.random.normal(
            loc=time_to_write_answer, scale=time_to_write_std_dev, size=number_samples
        ) + np.random.normal(
            loc=TIME_TO_MAKE_ANSWER[kind][0],
            scale=TIME_TO_MAKE_ANSWER[kind][1],
            size=number_samples,
        )


def time_to_answer_form(processed_fields, normalized_fields) -> Tuple[float, float]:
    """
    Provide an estimate of how long it would take an average user to respond to the questions
    on the provided form.
    We use signals such as the field type, name, and space provided for the response to come up with a
    rough estimate, based on whether the field is:
    1. fill in the blank
    2. gathered - e.g., an id number, case number, etc.
    3. third party: need to actually ask someone the information - e.g., income of not the user, anything else?
    4. created:
        a. short created (3 lines or so?)
        b. long created (anything over 3 lines)
    """
    field_answer_time_simulators: List[Callable[[int], np.ndarray]] = []
    for index, field in enumerate(processed_fields):
        field_answer_time_simulators.append(
            time_to_answer_field(field, normalized_fields[index])
        )
    # Run a monte carlo simulation to get times to answer and standard deviation
    num_samples = 20000
    np_array = np.zeros(num_samples)
    for field_simulator in field_answer_time_simulators:
        np_array += field_simulator(num_samples)
    return sigfig.round(np_array.mean(), 2), sigfig.round(np_array.std(), 2)


def cleanup_text(text: str, fields_to_sentences: bool = False) -> str:
    """
    Apply cleanup routines to text to provide more accurate readability statistics.
    """
    # Replace \n with .
    text = re.sub(r"(\n|\r)+", ". ", text)
    # Replace non-punctuation characters with " "
    text = re.sub(r"[^\w.,;!?@'\"“”‘’'″‶ ]", " ", text)
    # _ is considered a word character, remove it
    text = re.sub(r"_+", " ", text)
    if fields_to_sentences:
        # Turn : into . (so fields are treated as one sentence)
        text = re.sub(r":", ".", text)
    # Condense repeated " "
    text = re.sub(r" +", " ", text)
    # Remove any sentences that are just composed of a space
    text = re.sub(r"\. +\.", ". ", text)
    # Remove any repeated .
    text = re.sub(r"\.+", ".", text)
    # Remove space before final period
    text = re.sub(r" \.", ".", text)
    return text


def all_caps_words(text: str) -> int:
    results = re.findall(r"([A-Z][A-Z]+)", text)
    if results:
        return len(results)
    return 0


def text_complete(
    system_message: str,
    user_message: Optional[str] = None,
    max_tokens: int = 500,
    creds: Optional[OpenAiCreds] = None,
    temperature: float = 0,
    api_key: Optional[str] = None,
    model: str = "gpt-5-nano",
    # Legacy parameter for backward compatibility
    prompt: Optional[str] = None,
) -> Union[str, Dict]:
    """Run a prompt via openAI's API and return the result.

    Args:
        system_message (str): The system message that sets the context/role for the AI.
        user_message (Optional[str]): The user message/question. If None, system_message is used as the prompt.
        max_tokens (int, optional): The number of tokens to generate. Defaults to 500.
        creds (Optional[OpenAiCreds], optional): The credentials to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0. Note: Not supported by GPT-5 family models.
        api_key (Optional[str], optional): Explicit API key to use. Defaults to None.
        model (str, optional): The model to use. Defaults to "gpt-5-nano".
        prompt (Optional[str]): Legacy parameter for backward compatibility. If provided, used as system message.
    
    Returns:
        Union[str, Dict]: Returns a parsed dictionary if JSON was requested and successfully parsed, 
                         otherwise returns the raw string response.
    """
    # Handle backward compatibility
    if prompt is not None:
        system_message = prompt
        user_message = None
    
    # Resolve the API key using our helper function
    resolved_key = get_openai_api_key_from_sources(api_key, dict(creds) if creds else None)

    if resolved_key:
        openai_client = OpenAI(
            api_key=resolved_key, organization=creds.get("org") if creds else None
        )
    elif creds:
        openai_client = OpenAI(api_key=creds["key"], organization=creds["org"])
    else:
        if client:
            openai_client = client
        else:
            raise Exception("No OpenAI credentials provided")
    
    try:
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
        if user_message:
            messages.append({"role": "user", "content": user_message})
        
        # GPT-5 family models use different parameters
        completion_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        
        # GPT-5 models use max_completion_tokens instead of max_tokens and need more tokens due to reasoning
        if model.startswith("gpt-5"):
            # Increase tokens for GPT-5 models but respect the 128K completion token limit
            requested_tokens = min(max_tokens * 5, 128000)  # 5x multiplier but capped at 128K
            completion_params["max_completion_tokens"] = requested_tokens
        else:
            completion_params["max_tokens"] = max_tokens
            completion_params["temperature"] = temperature
        
        # Enable JSON mode if "json" appears in the prompt (case-insensitive)
        combined_prompt = system_message + (user_message or "")
        is_json_request = "json" in combined_prompt.lower()
        if is_json_request:
            completion_params["response_format"] = {"type": "json_object"}
        
        response = openai_client.chat.completions.create(**completion_params)
        raw_response = str((response.choices[0].message.content or "").strip())
        
        # If JSON was requested, try to parse and return the JSON object
        if is_json_request:
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError as e:
                # If JSON parsing fails, try to extract JSON from the response
                # Sometimes the model might include extra text around the JSON
                json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"Failed to parse JSON response: {e}")
                        return raw_response
                else:
                    print(f"No valid JSON found in response: {raw_response}")
                    return raw_response
        
        return raw_response
    except Exception as ex:
        print(f"{ex}")
        return "ApiError"


def complete_with_command(
    text,
    command,
    tokens,
    creds: Optional[OpenAiCreds] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """Combines some text with a command to send to open ai."""
    # For GPT-5-nano: 400K input token limit, so we can handle much larger inputs
    # Support up to 30 pages of PDF text (~300K characters = ~75K tokens)
    # Reserve space for command and response tokens: 375K - command - response = ~300K for text
    max_input_tokens = 300000  # Conservative limit for input text
    model_to_use = model or "gpt-5-nano"

    available_tokens = max_input_tokens - _token_count(command, model_to_use) - tokens
    max_length = max(available_tokens, 1)

    text = _truncate_to_token_limit(text, max_length, model_to_use)
    
    result = text_complete(
        system_message=command,
        user_message=text,
        max_tokens=tokens,
        creds=creds,
        api_key=api_key,
        model=model_to_use,
    )
    # Ensure we always return a string for this function
    if isinstance(result, dict):
        return json.dumps(result)
    return result


def plain_lang(text, creds: Optional[OpenAiCreds] = None) -> str:
    model = "gpt-5-nano"
    tokens = max(_token_count(text, model), 500)  # Minimum 500 tokens for GPT-5
    command = _load_prompt("plain_language")
    return complete_with_command(text, command, tokens, creds=creds, model=model)


def guess_form_name(
    text, creds: Optional[OpenAiCreds] = None, api_key: Optional[str] = None
) -> str:
    command = _load_prompt("guess_form_name")
    model = "gpt-5-nano"
    return complete_with_command(
        text, command, 200, creds=creds, api_key=api_key, model=model
    )


def describe_form(
    text, creds: Optional[OpenAiCreds] = None, api_key: Optional[str] = None
) -> str:
    command = _load_prompt("describe_form")
    model = "gpt-5-nano"
    return complete_with_command(
        text,
        command,
        3000,
        creds=creds,
        api_key=api_key,
        model=model,
    )  # Increased for more detailed descriptions


def needs_calculations(text: str) -> bool:
    # since we reomved SpaCy we can't use Doc,
    # so I rewrote this to provide similar functionality absent Doc
    # old code is commented out
    # def needs_calculations(text: Union[str, Doc]) -> bool:
    """A conservative guess at if a given form needs the filler to make math calculations,
    something that should be avoided. If"""
    CALCULATION_WORDS = ["subtract", "total", "minus", "multiply" "divide"]
    # if isinstance(text, str):
    #    doc = nlp(text)
    # else:
    #    doc = text
    # for token in doc:
    #    if token.text.lower() in CALCULATION_WORDS:
    #        return True
    for word in CALCULATION_WORDS:
        if word in text.lower():
            return True

    # TODO(brycew): anything better than a binary yes-no value on this?
    return False


def get_passive_sentences(
    text: Union[List, str],
    tools_token: Optional[str] = None,
    model: str = "gpt-5-nano",
    api_key: Optional[str] = None,
) -> List[Tuple[str, List[Tuple[int, int]]]]:
    """Return passive voice fragments for each sentence in ``text``.

    The function relies on OpenAI's language model (via ``passive_voice_detection``)
    to detect passive constructions. ``tools_token`` is kept for backward compatibility
    but is no longer used.

    Args:
        text (Union[List, str]): The input text or list of texts to analyze.
        tools_token (Optional[str], optional): Deprecated. Previously used for authentication with
            tools.suffolklitlab.org. Defaults to None.
        model (str, optional): The OpenAI model to use for detection. Defaults to "gpt-5-nano".
        api_key (Optional[str], optional): OpenAI API key to use. If None, will try docassemble
            config (if available) then environment variables. Defaults to None.
    Returns:
        List[Tuple[str, List[Tuple[int, int]]]]: A list of tuples, each containing the original text
            and a list of tuples representing the start and end positions of detected passive voice fragments.

    Note:
        At least for now, the fragment detection is no longer meaningful (except in tokenized sentences) because
        the LLM detection simply returns the full original sentence if it contains passive voice. We have not reimplemented
        this behavior of PassivePy.
    """
    if tools_token:
        pass  # deprecated

    sentences_with_highlights = []

    passive_voice_results = detect_passive_voice_segments(
        text,
        openai_client=client if client else None,
        model=model,
        api_key=api_key,
    )

    for item in passive_voice_results:
        for fragment in item[1]:
            sentences_with_highlights.append(
                (
                    item[0],
                    [
                        (match.start(), match.end())
                        for match in re.finditer(re.escape(fragment), item[0])
                    ],
                )
            )
    return sentences_with_highlights


def get_citations(text: str, tokenized_sentences: List[str]) -> List[str]:
    """
    Get citations and some extra surrounding context (the full sentence), if the citation is
    fewer than 5 characters (often eyecite only captures a section symbol
    for state-level short citation formats)
    """
    citations = eyecite.get_citations(
        eyecite.clean_text(text, ["all_whitespace", "underscores"])
    )
    citations_with_context = []
    tokens = set()
    for cite in citations:
        if len(cite.matched_text()) < 5:
            tokens.add(cite.matched_text())
        else:
            citations_with_context.append(cite.matched_text())
    for token in tokens:
        citations_with_context.extend(
            [sentence for sentence in tokenized_sentences if token in sentence]
        )

    return citations_with_context


# NOTE: omitting "CID" for Credit Card IDs since it has a lot of false positives.
FIELD_PATTERNS = {
    "Bank Account Number": [
        r"account[\W_]*number",
        r"ABA$",
        r"routing[\W_]*number",
        r"checking",
    ],
    "Credit Card Number": [r"credit[\W_]*card", r"(CV[CDV]2?|CCV|CSC)"],
    "Driver's License Number": [r"drivers[\W_]*license", r".?DL$"],
    "Social Security Number": [r"social[\W_]*security[\W_]*number", r"SSN", r"TIN$"],
}
FIELD_REGEXES = {
    name: re.compile("|".join(patterns), re.IGNORECASE | re.MULTILINE)
    for name, patterns in FIELD_PATTERNS.items()
}


def get_sensitive_data_types(
    fields: List[str], fields_old: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Given a list of fields, identify those related to sensitive information and return a dictionary with the sensitive
    fields grouped by type. A list of the old field names can also be provided. These fields should be in the same
    order. Passing the old field names allows the sensitive field algorithm to match more accurately. The return value
    will not contain the old field name, only the corresponding field name from the first parameter.

    The sensitive data types are: Bank Account Number, Credit Card Number, Driver's License Number, and Social Security
    Number.
    """

    if fields_old is not None and len(fields) != len(fields_old):
        raise ValueError(
            "If provided, fields_old must have the same number of items as fields."
        )

    sensitive_data_types: Dict[str, List[str]] = {}
    num_fields = len(fields)
    for i, field in enumerate(fields):
        for name, regex in FIELD_REGEXES.items():
            if re.search(regex, field):
                sensitive_data_types.setdefault(name, []).append(field)
            elif fields_old is not None and re.search(regex, fields_old[i]):
                sensitive_data_types.setdefault(name, []).append(field)
    return sensitive_data_types


def substitute_phrases(
    input_string: str, substitution_phrases: Dict[str, str]
) -> Tuple[str, List[Tuple[int, int]]]:
    """Substitute phrases in the input string and return the new string and positions of substituted phrases.

    Args:
        input_string (str): The input string containing phrases to be replaced.
        substitution_phrases (Dict[str, str]): A dictionary mapping original phrases to their replacement phrases.

    Returns:
        Tuple[str, List[Tuple[int, int]]]: A tuple containing the new string with substituted phrases and a list of
                                          tuples, each containing the start and end positions of the substituted
                                          phrases in the new string.

    Example:
        >>> input_string = "The quick brown fox jumped over the lazy dog."
        >>> substitution_phrases = {"quick brown": "swift reddish", "lazy dog": "sleepy canine"}
        >>> new_string, positions = substitute_phrases(input_string, substitution_phrases)
        >>> print(new_string)
        "The swift reddish fox jumped over the sleepy canine."
        >>> print(positions)
        [(4, 17), (35, 48)]
    """
    # Sort the substitution phrases by length in descending order
    sorted_phrases = sorted(
        substitution_phrases.items(), key=lambda x: len(x[0]), reverse=True
    )

    matches = []

    # Find all matches for the substitution phrases
    for original, replacement in sorted_phrases:
        for match in re.finditer(
            r"\b" + re.escape(original) + r"\b", input_string, re.IGNORECASE
        ):
            matches.append((match.start(), match.end(), replacement))

    # Sort the matches based on their starting position
    matches.sort(key=lambda x: x[0])

    new_string = ""
    substitutions: List[Tuple[int, int]] = []
    prev_end_pos = 0

    # Build the new string and substitutions list
    for start_pos, end_pos, replacement in matches:
        if start_pos >= prev_end_pos:
            new_string += input_string[prev_end_pos:start_pos] + replacement
            substitutions.append((len(new_string) - len(replacement), len(new_string)))
            prev_end_pos = end_pos

    new_string += input_string[prev_end_pos:]

    return new_string, substitutions


def substitute_neutral_gender(input_string: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Substitute gendered phrases with neutral phrases in the input string.
    Primary source is https://github.com/joelparkerhenderson/inclusive-language
    """
    with open(GENDERED_TERMS_PATH) as f:
        terms = yaml.safe_load(f)
    return substitute_phrases(input_string, terms)


def substitute_plain_language(input_string: str) -> Tuple[str, List[Tuple[int, int]]]:
    """
    Substitute complex phrases with simpler alternatives.
    Source of terms is drawn from https://www.plainlanguage.gov/guidelines/words/
    """
    with open(PLAIN_LANGUAGE_TERMS_PATH) as f:
        terms = yaml.safe_load(f)
    return substitute_phrases(input_string, terms)


def transformed_sentences(
    sentence_list: List[str], fun: Callable
) -> List[Tuple[str, str, List[Tuple[int, int]]]]:
    """
    Apply a function to a list of sentences and return only the sentences with changed terms.
    The result is a tuple of the original sentence, new sentence, and the starting and ending position
    of each changed fragment in the sentence.
    """
    transformed: List[Tuple[str, str, List[Tuple[int, int]]]] = []
    for sentence in sentence_list:
        run = fun(sentence)
        if run[0] != sentence:
            transformed.append((sentence, run[0], run[1]))
    return transformed


def fallback_rename_fields(field_names: List[str]) -> Tuple[List[str], List[float]]:
    """
    A simple fallback renaming scheme that just makes field names lowercase
    and replaces spaces with underscores.
    """
    length = len(field_names)
    last = "null"
    new_names = []
    new_names_conf = []
    for i, field_name in enumerate(field_names):
        new_name, new_confidence = normalize_name(
            "",
            "",
            i,
            i / length,
            last,
            field_name,
        )
        new_names.append(new_name)
        new_names_conf.append(new_confidence)
        last = field_name
    new_names = [
        v + "__" + str(new_names[:i].count(v) + 1) if new_names.count(v) > 1 else v
        for i, v in enumerate(new_names)
    ]
    return new_names, new_names_conf


def parse_form(
    in_file: str,
    title: Optional[str] = None,
    jur: Optional[str] = None,
    cat: Optional[str] = None,
    normalize: bool = True,
    spot_token: Optional[str] = None,
    tools_token: Optional[str] = None,
    openai_creds: Optional[OpenAiCreds] = None,
    openai_api_key: Optional[str] = None,
    rewrite: bool = False,
    debug: bool = False,
):
    """
    Read in a pdf, pull out basic stats, attempt to normalize its form fields, and re-write the
    in_file with the new fields (if `rewrite=1`). If you pass a spot token, we will guess the
    NSMI code. If you pass openai creds, we will give suggestions for the title and description.
    If you pass openai_api_key, it will be used for passive voice detection (overrides creds and env vars).

    Args:
        in_file: the path to the PDF file to analyze
        title: the title of the form, if not provided we will try to guess it
        jur: the jurisdiction to use for normalization (e.g., "ny" or "ca")
        cat: the category to use for normalization (e.g., "divorce" or "small_claims")
        normalize: whether to normalize the field names
        spot_token: the token to use for spot.suffolklitlab.org, if provided we will
            attempt to guess the NSMI code
        tools_token: the token to use for tools.suffolklitlab.org, needed for normalization
        openai_creds: the OpenAI credentials to use, if provided we will attempt to
            guess the title and description
        openai_api_key: an explicit OpenAI API key to use, if provided it will override
            any creds or environment variables
        rewrite: whether to rewrite the PDF in place with the new field names
        debug: whether to print debug information

    Returns: a dictionary of information about the form
    """
    unlock_pdf_in_place(in_file)
    the_pdf = pikepdf.open(in_file)
    pages_count = len(the_pdf.pages)

    try:
        with time_limit(15):
            all_fields_per_page = get_existing_pdf_fields(the_pdf)
            ff = []
            for fields_in_page in all_fields_per_page:
                ff.extend(fields_in_page)
    except TimeoutException as e:
        print("Timed out!")
        ff = None
    except AttributeError:
        ff = None
    # Resolve API key once at the top for consistency
    resolved_api_key = get_openai_api_key_from_sources(
        openai_api_key, dict(openai_creds) if openai_creds else None
    )

    field_names = [field.name for field in ff] if ff else []
    f_per_page = len(field_names) / pages_count
    # some PDFs (698c6784e6b9b9518e5390fd9ec31050) have vertical text, but it's not detected.
    # Text contains a bunch of "(cid:72)", garbage output (reading level is like 1000).
    # Our workaround is to ask GPT3 if it looks like a court form, and if not, try running
    # ocrmypdf.
    original_text = extract_text(in_file, laparams=LAParams(detect_vertical=True))
    text = cleanup_text(original_text)
    description = (
        describe_form(text, creds=openai_creds, api_key=resolved_api_key)
        if (openai_creds or resolved_api_key)
        else ""
    )
    try:
        readability = (
            textstat.text_standard(text, float_output=True)  # type: ignore[attr-defined]
            if text
            else -1
        )
    except:
        readability = -1
    # Still attempt to re-evaluate if not using openai
    if (
        not original_text
        or (openai_creds and description == "abortthisnow.")
        or readability > 30
    ):
        # We do not care what the PDF output is, doesn't add that much time
        ocr_p = [
            "ocrmypdf",
            "--force-ocr",
            "--rotate-pages",
            "--sidecar",
            "-",
            in_file,
            "/tmp/test.pdf",
        ]
        process = subprocess.run(ocr_p, timeout=60, check=False, capture_output=True)
        if process.returncode == 0:
            original_text = process.stdout.decode()
            text = cleanup_text(original_text)
            try:
                readability = (
                    textstat.text_standard(text, float_output=True)  # type: ignore[attr-defined]
                    if text
                    else -1
                )
            except:
                readability = -1

    new_title = (
        guess_form_name(text, creds=openai_creds, api_key=resolved_api_key)
        if (openai_creds or resolved_api_key)
        else ""
    )
    if not title:
        if hasattr(the_pdf.docinfo, "Title"):
            title = str(the_pdf.docinfo.Title)
        if (
            not title
            and new_title
            and (new_title != "ApiError" and new_title.lower() != "abortthisnow.")
        ):
            title = new_title
        if not title or title == "ApiError" or title.lower() == "abortthisnow.":
            fallback_title: Optional[str] = None
            matches = re.search(r"(.*)\n", text)
            if matches and matches.group(1).strip():
                fallback_title = matches.group(1).strip()

            if not fallback_title and original_text:
                sentences = split_sentences(original_text)
                if sentences:
                    fallback_title = sentences[0].strip()

            if not fallback_title and text:
                cleaned_sentences = [
                    segment.strip()
                    for segment in re.split(r"[.!?]", text)
                    if segment.strip()
                ]
                if cleaned_sentences:
                    fallback_title = cleaned_sentences[0]

            def _clean_title_candidate(raw: str) -> str:
                candidate = re.sub(r"\s+", " ", raw).strip()
                candidate = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", candidate)
                candidate = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", candidate)
                return candidate.strip(" ._-")

            def _looks_reasonable(candidate: str) -> bool:
                tokens = candidate.split()
                if not tokens:
                    return False
                single_letter_tokens = [
                    token for token in tokens if len(token) == 1 and token.lower() not in {"a", "i"}
                ]
                if len(single_letter_tokens) > max(1, len(tokens) // 3):
                    return False
                if len(candidate) > 160:
                    return False
                if re.search(r"[A-Z]{2,}(AND|OR|OF|FOR)[A-Z]", candidate):
                    return False
                return True

            if fallback_title:
                fallback_title = _clean_title_candidate(fallback_title)
                if not _looks_reasonable(fallback_title):
                    fallback_title = ""

            if not fallback_title:
                fallback_title = re_case(Path(in_file).stem.replace("_", " ")).title()

            title = fallback_title if fallback_title else "(Untitled)"
    nsmi = spot(title + ". " + text, token=spot_token) if spot_token else []
    if normalize:
        if (openai_creds or resolved_api_key) and field_names:
            try:
                field_mappings = rename_pdf_fields_with_context(
                    in_file,
                    field_names,
                    openai_creds=openai_creds,
                    api_key=resolved_api_key,
                )
                new_names = [field_mappings.get(name, name) or name for name in field_names]
                new_names_conf = [0.8 if field_mappings.get(name) else 0.1 for name in field_names]
            except Exception as e:
                print(f"LLM field renaming failed: {e}, falling back to traditional approach")
                # Fallback to traditional approach
                new_names, new_names_conf = fallback_rename_fields(field_names)
        else:
            new_names, new_names_conf = fallback_rename_fields(field_names)
    else:
        new_names = field_names
        new_names_conf = []

    tokenized_sentences = split_sentences(original_text)
    # No need to detect passive voice in very short sentences
    sentences = [s for s in tokenized_sentences if len(s.split(" ")) > 2]

    try:
        passive_sentences = get_passive_sentences(sentences, api_key=resolved_api_key)
        passive_sentences_count = len(passive_sentences)
    except ValueError:
        passive_sentences_count = 0
        passive_sentences = []

    citations = get_citations(original_text, tokenized_sentences)
    plain_language_suggestions = transformed_sentences(
        sentences, substitute_plain_language
    )
    neutral_gender_suggestions = transformed_sentences(
        sentences, substitute_neutral_gender
    )
    word_count = len(text.split(" "))
    all_caps_count = all_caps_words(text)
    field_types = field_types_and_sizes(ff)
    classified = [
        classify_field(field, new_names[index])
        for index, field in enumerate(field_types)
    ]
    sensitive_data_types = get_sensitive_data_types(new_names, field_names)

    slotin_count = sum(1 for c in classified if c == AnswerType.SLOT_IN)
    gathered_count = sum(1 for c in classified if c == AnswerType.GATHERED)
    third_party_count = sum(1 for c in classified if c == AnswerType.THIRD_PARTY)
    created_count = sum(1 for c in classified if c == AnswerType.CREATED)
    sentence_count = sum(1 for _ in sentences)
    field_count = len(field_names)
    difficult_words = textstat.difficult_words_list(text)  # type: ignore[attr-defined]
    difficult_word_count = len(difficult_words)
    citation_count = len(citations)
    pdf_is_tagged = is_tagged(the_pdf)
    stats = {
        "title": title,
        "suggested title": new_title,
        "description": description,
        "category": cat,
        "pages": pages_count,
        "reading grade level": readability,
        "time to answer": (
            time_to_answer_form(field_types_and_sizes(ff), new_names)
            if ff
            else [-1, -1]
        ),
        "list": nsmi,
        "avg fields per page": f_per_page,
        "fields": new_names,
        "fields_conf": new_names_conf,
        "fields_old": field_names,
        "sensitive data types": sensitive_data_types,
        "text": text,
        "original_text": original_text,
        "number of sentences": sentence_count,
        "sentences per page": sentence_count / pages_count,
        "number of passive voice sentences": passive_sentences_count,
        "passive sentences": passive_sentences,
        "number of all caps words": all_caps_count,
        "citations": citations,
        "total fields": field_count,
        "slotin percent": slotin_count / field_count if field_count > 0 else 0,
        "gathered percent": gathered_count / field_count if field_count > 0 else 0,
        "created percent": created_count / field_count if field_count > 0 else 0,
        "third party percent": (
            third_party_count / field_count if field_count > 0 else 0
        ),
        "passive voice percent": (
            passive_sentences_count / sentence_count if sentence_count > 0 else 0
        ),
        "citations per field": citation_count / field_count if field_count > 0 else 0,
        "citation count": citation_count,
        "all caps percent": all_caps_count / word_count,
        "normalized characters per field": (
            sum(get_adjusted_character_count(field) for field in field_types)
            / field_count
            if ff
            else 0
        ),
        "difficult words": difficult_words,
        "difficult word count": difficult_word_count,
        "difficult word percent": difficult_word_count / word_count,
        "calculation required": needs_calculations(text),
        "plain language suggestions": plain_language_suggestions,
        "neutral gender suggestions": neutral_gender_suggestions,
        "pdf_is_tagged": pdf_is_tagged,
    }
    if debug and ff:
        debug_fields = []
        for index, field in enumerate(field_types_and_sizes(ff)):
            debug_fields.append(
                {
                    "name": field["var_name"],
                    "input type": str(field["type"]),
                    "max length": field["max_length"],
                    "inferred answer type": str(
                        classify_field(field, new_names[index])
                    ),
                    "time to answer": list(
                        time_to_answer_field(field, new_names[index])(1)
                    ),
                }
            )
        stats["debug fields"] = debug_fields
    if rewrite:
        try:
            my_pdf = pikepdf.Pdf.open(in_file, allow_overwriting_input=True)
            fields_too = (
                my_pdf.Root.AcroForm.Fields
            )  # [0]["/Kids"][0]["/Kids"][0]["/Kids"][0]["/Kids"]
            # print(repr(fields_too))
            for k, field_name in enumerate(new_names):
                # print(k,field)
                fields_too[k].T = re.sub(r"^\*", "", field_name)
            my_pdf.save(in_file)
            my_pdf.close()
        except Exception as ex:
            stats["error"] = f"could not change form fields: {ex}"
    return stats


def _form_complexity_per_metric(stats):
    # check for fields that require user to look up info, when found add to complexity
    # maybe score these by minutes to recall/fill out
    # so, figure out words per minute, mix in with readability and page number and field numbers

    # TODO(brycew):
    # to write: options with unknown?
    # to write: fields with exact info
    # to write: fields with open ended responses (text boxes)
    metrics = [
        {"name": "reading grade level", "weight": 10 / 7, "intercept": 5},
        {"name": "calculation required", "weight": 2},
        # {"name": "time to answer", "weight": 2},
        {"name": "pages", "weight": 2},
        {"name": "citations per field", "weight": 1.2},
        {"name": "avg fields per page", "weight": 1 / 8},
        {"name": "normalized characters per field", "weight": 1 / 8},
        {"name": "sentences per page", "weight": 0.05},
        # percents will have a higher weight, because they are between 0 and 1
        {"name": "slotin percent", "weight": 2},
        {"name": "gathered percent", "weight": 5},
        {"name": "third party percent", "weight": 10},
        {"name": "created percent", "weight": 20},
        {"name": "passive voice percent", "weight": 4},
        {"name": "all caps percent", "weight": 10},
        {"name": "difficult word percent", "weight": 15},
    ]

    def weight(stats, metric):
        """Handles if we need to scale / "normalize" the metrics at all."""
        name = metric["name"]
        weight = metric.get("weight") or 1
        val = 0
        if "clip" in metric:
            val = min(max(stats.get(name, 0), metric["clip"][0]), metric["clip"][1])
        elif isinstance(stats.get(name), bool):
            val = 1 if stats.get(name) else 0
        else:
            val = stats.get(name, 0)
        if "intercept" in metric:
            val -= metric["intercept"]
        return val * weight

    return [(m["name"], stats[m["name"]], weight(stats, m)) for m in metrics]


def form_complexity(stats):
    """Gets a single number of how hard the form is to complete. Higher is harder."""
    metrics = _form_complexity_per_metric(stats)
    return sum(val[2] for val in metrics)
