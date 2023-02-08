# Updated on 2022-12-12

import os
import re
import subprocess
from sklearn.metrics import classification_report
import spacy
from spacy.tokens import Doc
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams

import pikepdf
import textstat
import requests
import json
import networkx as nx
import numpy as np
import pandas as pd
from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from joblib import load
import nltk
from nltk.tokenize import sent_tokenize
from PassivePySrc import PassivePy
import eyecite
from enum import Enum
import sigfig
from .pdf_wrangling import (
    get_existing_pdf_fields,
    FormField,
    FieldType,
    unlock_pdf_in_place,
)

try:
    from nltk.corpus import stopwords

    stopwords.words
except:
    print("Downloading stopwords")
    nltk.download("stopwords")
    from nltk.corpus import stopwords
try:
    nltk.data.find("tokenizers/punkt")
except:
    nltk.download("punkt")

import math
from contextlib import contextmanager
import threading
import _thread
from typing import (
    Optional,
    Union,
    Iterable,
    List,
    Dict,
    Tuple,
    Callable,
    TypedDict,
)

import openai
from transformers import GPT2TokenizerFast

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

stop_words = set(stopwords.words("english"))

try:
    # this takes a while to load
    import en_core_web_md

    nlp = en_core_web_md.load()
except:
    print("Downloading word2vec model en_core_web_md")
    import subprocess

    bashCommand = "python -m spacy download en_core_web_md"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f"output of word2vec model download: {str(output)}")
    import en_core_web_md

    nlp = en_core_web_md.load()

passivepy = PassivePy.PassivePyAnalyzer(nlp=nlp)


# Load local variables, models, and API key(s).

included_fields = load(
    os.path.join(os.path.dirname(__file__), "data", "included_fields.joblib")
)
jurisdictions = load(
    os.path.join(os.path.dirname(__file__), "data", "jurisdictions.joblib")
)
groups = load(os.path.join(os.path.dirname(__file__), "data", "groups.joblib"))
clf_field_names = load(
    os.path.join(os.path.dirname(__file__), "data", "clf_field_names.joblib")
)
with open(
    os.path.join(os.path.dirname(__file__), "keys", "spot_token.txt"), "r"
) as in_file:
    default_spot_token = in_file.read().rstrip()
with open(
    os.path.join(os.path.dirname(__file__), "keys", "openai_org.txt"), "r"
) as in_file:
    openai.organization = in_file.read().rstrip()
with open(
    os.path.join(os.path.dirname(__file__), "keys", "openai_key.txt"), "r"
) as in_file:
    openai.api_key = in_file.read().rstrip()

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
        "text": text,
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
        ["^((My|Your|Full( legal)?) )?Name$", "users1_name"],
        ["^(Typed or )?Printed Name\s?\d*$", "users1_name"],
        ["^(DOB|Date of Birth|Birthday)$", "users1_birthdate"],
        ## Address
        ["^(Street )?Address$", "users1_address_line_one"],
        ["^City State Zip$", "users1_address_line_two"],
        ["^City$", "users1_address_city"],
        ["^State$", "users1_address_state"],
        ["^Zip( Code)?$", "users1_address_zip"],
        ## Contact
        ["^(Phone|Telephone)$", "users1_phone_number"],
        ["^Email( Address)$", "users1_email"],
        # Parties
        ["^plaintiff\(?s?\)?$", "plaintiff1_name"],
        ["^defendant\(?s?\)?$", "defendant1_name"],
        ["^petitioner\(?s?\)?$", "petitioners1_name"],
        ["^respondent\(?s?\)?$", "respondents1_name"],
        # Court info
        ["^(Court\s)?Case\s?(No|Number)?\s?A?$", "docket_number"],
        ["^file\s?(No|Number)?\s?A?$", "docket_number"],
        # Form info
        ["^(Signature|Sign( here)?)\s?\d*$", "users1_signature"],
        ["^Date\s?\d*$", "signature_date"],
    ]
    for regex in regex_list:
        text = re.sub(regex[0], regex[1], text, flags=re.IGNORECASE)
    return text


def reformat_field(text: str, max_length: int = 30):
    """
    Transforms a string of text into a snake_case variable close in length to `max_length` name by
    summarizing the string and stitching the summary together in snake_case.
    h/t https://towardsdatascience.com/nlp-building-a-summariser-68e0c19e3a93
    """
    orig_title = text.lower()
    orig_title = re.sub("[^a-zA-Z]+", " ", orig_title)
    orig_title_words = orig_title.split()
    deduped_sentence = []
    for word in orig_title_words:
        if word not in deduped_sentence:
            deduped_sentence.append(word)
    filtered_sentence = [w for w in deduped_sentence if not w.lower() in stop_words]
    filtered_title_words = filtered_sentence
    characters = len(" ".join(filtered_title_words))
    if characters > 0:
        words = len(filtered_title_words)
        av_word_len = math.ceil(
            len(" ".join(filtered_title_words)) / len(filtered_title_words)
        )
        x_words = math.floor((max_length) / av_word_len)
        sim_mat = np.zeros([len(filtered_title_words), len(filtered_title_words)])
        # for each word compared to other
        for i in range(len(filtered_title_words)):
            for j in range(len(filtered_title_words)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(
                        nlp(filtered_title_words[i]).vector.reshape(1, 300),
                        nlp(filtered_title_words[j]).vector.reshape(1, 300),
                    )[0, 0]
        try:
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            sorted_scores = sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            )
            if x_words > len(scores):
                x_words = len(scores)
            i = 0
            new_title = ""
            for x in filtered_title_words:
                if scores[i] >= sorted_scores[x_words - 1][1]:
                    if len(new_title) > 0:
                        new_title += "_"
                    new_title += x
                i += 1
            return new_title
        except:
            return "_".join(filtered_title_words)
    else:
        if re.search("^(\d+)$", text):
            return "unknown"
        else:
            return re.sub("\s+", "_", text.lower())


def norm(row):
    """Normalize a word vector."""
    try:
        matrix = row.reshape(1, -1).astype(np.float64)
        return normalize(matrix, axis=1, norm="l1")[0]
    except Exception as e:
        print("===================")
        print("Error: ", e)
        print("===================")
        return np.NaN


def vectorize(text: str, normalize: bool = True):
    """Vectorize a string of text."""
    output = nlp(str(text)).vector
    if normalize:
        return norm(output)
    else:
        return output


# Given an auto-generated field name and context from the form where it appeared, this function attempts to normalize the field name. Here's what's going on:
# 1. It will `re_case` the variable text
# 2. Then it will run the output through `regex_norm_field`
# 3. If it doesn't find anything, it will use the ML model `clf_field_names`
# 4. If the prediction isn't very confident, it will run it through `reformat_field`


def normalize_name(
    jur: str, group: str, n: int, per, last_field: str, this_field: str
) -> Tuple[str, float]:
    """Add hard coded conversions maybe by calling a function
    if returns 0 then fail over to ML or other way around poor prob -> check hard-coded.
    Retuns the new name and a confidence value between 0 and 1"""
    if this_field not in included_fields:
        this_field = re_case(this_field)
        out_put = regex_norm_field(this_field)
        conf = 1.0
        if out_put == this_field:
            params = []
            for item in jurisdictions:
                if jur == item:
                    params.append(1)
                else:
                    params.append(0)
            for item in groups:
                if group == item:
                    params.append(1)
                else:
                    params.append(0)
            params.append(n)
            params.append(per)
            for vec in vectorize(this_field):
                params.append(vec)
            for item in included_fields:
                if last_field == item:
                    params.append(1)
                else:
                    params.append(0)
            pred = clf_field_names.predict([params])
            prob = clf_field_names.predict_proba([params])
            conf = prob[0].tolist()[prob[0].tolist().index(max(prob[0].tolist()))]
            out_put = pred[0]
    else:
        out_put = this_field
        conf = 1
    if out_put in included_fields:
        if conf >= 0:
            return (
                "*" + out_put,
                conf,
            )  # this * is a hack to show when something is in the list of known fields later. I need to fix this
        else:
            return reformat_field(this_field), conf
    else:
        return reformat_field(this_field), conf


# Take a list of AL variables and spits out suggested groupings. Here's what's going on:
#
# 1. It reads in a list of fields (e.g., `["user_name","user_address"]`)
# 2. Splits each field into words (e.g., turning `user_name` into `user name`)
# 3. It then turns these ngrams/"sentences" into vectors using word2vec.
# 4. For the collection of fields, it finds clusters of these "sentences" within the semantic space defined by word2vec. Currently it uses Affinity Propagation. See https://machinelearningmastery.com/clustering-algorithms-with-python/


def cluster_screens(fields: List[str] = [], damping: float = 0.7):
    """Takes in a list (fields) and returns a suggested screen grouping
    Set damping to value >= 0.5 or < 1 to tune how related screens should be"""
    vec_mat = np.zeros([len(fields), 300])
    for i in range(len(fields)):
        vec_mat[i] = [nlp(re_case(fields[i])).vector][0]
    # create model
    model = AffinityPropagation(damping=damping, random_state=None)
    # model = AffinityPropagation(damping=damping,random_state=4) consider using this to get consistent results. note will have to require newer version
    # fit the model
    model.fit(vec_mat)
    # assign a cluster to each example
    yhat = model.predict(vec_mat)
    # retrieve unique clusters
    clusters = unique(yhat)
    screens = {}
    # sim = np.zeros([5,300])
    for i, cluster in enumerate(clusters):
        this_screen = where(yhat == cluster)[0]
        vars = []
        for screen in this_screen:
            # sim[screen]=vec_mat[screen] # use this spot to add up vectors for compare to list
            vars.append(fields[screen])
        screens["screen_%s" % i] = vars
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
        # We chunk answers into three different lengths rather than directly using the character count,
        # as forms can give very different spaces for the same data without regard to the room the
        # user actually needs. But small, medium, and full page is still helpful information.
        ONE_WORD = 4.7  # average word length: https://www.researchgate.net/figure/Average-word-length-in-the-English-language-Different-colours-indicate-the-results-for_fig1_230764201
        ONE_LINE = 115  # Standard line is ~ 115 characters wide at 12 point font
        SHORT_ANSWER = (
            ONE_LINE * 2
        )  # Anything over 1 line but less than 3 probably needs about the same time to answer
        MEDIUM_ANSWER = ONE_LINE * 5
        LONG_ANSWER = (
            ONE_LINE * 10
        )  # Anything over 10 lines probably needs a full page but form author skimped on space
        if field["max_length"] <= ONE_LINE or (
            field["max_length"] <= ONE_LINE * 2 and kind == AnswerType.SLOT_IN
        ):
            time_to_write_answer = ONE_WORD * 2 / cpm
            time_to_write_std_dev = ONE_WORD * 2 / cpm_std_dev
        elif field["max_length"] <= SHORT_ANSWER:
            time_to_write_answer = SHORT_ANSWER / cpm
            time_to_write_std_dev = SHORT_ANSWER / cpm_std_dev
        elif field["max_length"] <= MEDIUM_ANSWER:
            time_to_write_answer = MEDIUM_ANSWER / cpm
            time_to_write_std_dev = MEDIUM_ANSWER / cpm_std_dev
        else:
            time_to_write_answer = LONG_ANSWER / cpm
            time_to_write_std_dev = LONG_ANSWER / cpm_std_dev
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


class OpenAiCreds(TypedDict):
    org: str
    key: str


def text_complete(prompt, max_tokens=500, creds: Optional[OpenAiCreds] = None) -> str:
    if creds:
        openai.organization = creds["org"].strip()
        openai.api_key = creds["key"].strip()

    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=max_tokens,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
        )
        return str(response["choices"][0]["text"].strip())
    except Exception as ex:
        print(f"{ex}")
        return "ApiError"


def complete_with_command(
    text, command, tokens, creds: Optional[OpenAiCreds] = None
) -> str:
    """Combines some text with a command to send to open ai."""
    # OpenAI's max number of tokens length is 4097, so we trim the input text to 4080 - command - tokens length.
    # A bit less than 4097 in case the tokenizer is wrong
    # don't deal with negative numbers, clip at 1 (OpenAi will error anyway)
    max_length = max(4080 - len(tokenizer(command)["input_ids"]) - tokens, 1)
    text_tokens = tokenizer(text)
    if len(text_tokens["input_ids"]) > max_length:
        text = tokenizer.decode(
            tokenizer(text, truncation=True, max_length=max_length)["input_ids"]
        )
    return text_complete(text + "\n\n" + command, max_tokens=tokens, creds=creds)


def plain_lang(text, creds: Optional[OpenAiCreds] = None) -> str:
    tokens = len(tokenizer(text)["input_ids"])
    command = "Rewrite the above at a sixth grade reading level."
    return complete_with_command(text, command, tokens, creds=creds)


def guess_form_name(text, creds: Optional[OpenAiCreds] = None) -> str:
    command = 'If the above is a court form, write the form\'s name, otherwise respond with the word "abortthisnow.".'
    return complete_with_command(text, command, 20, creds=creds)


def describe_form(text, creds: Optional[OpenAiCreds] = None) -> str:
    command = 'If the above is a court form, write a brief description of its purpose at a sixth grade reading level, otherwise respond with the word "abortthisnow.".'
    return complete_with_command(text, command, 250, creds=creds)


def needs_calculations(text: Union[str, Doc]) -> bool:
    """A conservative guess at if a given form needs the filler to make math calculations,
    something that should be avoided. If"""
    CALCULATION_WORDS = ["subtract", "total", "minus", "multiply" "divide"]
    if isinstance(text, str):
        doc = nlp(text)
    else:
        doc = text
    for token in doc:
        if token.text.lower() in CALCULATION_WORDS:
            return True
    # TODO(brycew): anything better than a binary yes-no value on this?
    return False


def get_passive_sentences(text: Union[List, str]) -> List[Tuple[str, List[str]]]:
    """Return a list of tuples, where each tuple represents a
    sentence in which passive voice was detected along with a list of each
    fragment that is phrased in the passive voice. The combination of the two
    can be used in the PDFStats frontend to highlight the passive text in an
    individual sentence.

    Text can either be a string or a list of strings.
    If provided a single string, it will be tokenized with NTLK and
    sentences containing fewer than 2 words will be ignored.
    """
    # Sepehri, A., Markowitz, D. M., & Mir, M. (2022, February 3).
    # PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data. Retrieved from psyarxiv.com/bwp3t

    if isinstance(text, str):
        sentences = [s for s in sent_tokenize(text) if len(s.split(" ")) > 2]
        if not sentences:
            raise ValueError(
                "There are no sentences over 2 words in the provided text."
            )
    elif isinstance(text, list):
        sentences = text
    else:
        raise ValueError(f"Can't tokenize {type(text)} object into sentences")

    passive_text_df = passivepy.match_corpus_level(pd.DataFrame(sentences), 0)
    matching_rows = passive_text_df[passive_text_df["binary"] > 0]

    return list(zip(matching_rows["document"], matching_rows["all_passives"]))


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


def parse_form(
    in_file: str,
    title: Optional[str] = None,
    jur: Optional[str] = None,
    cat: Optional[str] = None,
    normalize: bool = True,
    spot_token: Optional[str] = None,
    openai_creds: Optional[OpenAiCreds] = None,
    rewrite: bool = False,
    debug: bool = False,
):
    """
    Read in a pdf, pull out basic stats, attempt to normalize its form fields, and re-write the
    in_file with the new fields (if `rewrite=1`). If you pass a spot token, we will guess the
    NSMI code. If you pass openai creds, we will give suggestions for the title and description.
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
    field_names = [field.name for field in ff] if ff else []
    f_per_page = len(field_names) / pages_count
    # some PDFs (698c6784e6b9b9518e5390fd9ec31050) have vertical text, but it's not detected.
    # Text contains a bunch of "(cid:72)", garbage output (reading level is like 1000).
    # Our workaround is to ask GPT3 if it looks like a court form, and if not, try running
    # ocrmypdf.
    original_text = extract_text(in_file, laparams=LAParams(detect_vertical=True))
    text = cleanup_text(original_text)
    description = describe_form(text, creds=openai_creds) if openai_creds else ""
    try:
        readability = textstat.text_standard(text, float_output=True) if text else -1
    except:
        readability = -1
    # Still attempt to re-evaluate if not using openai
    if (openai_creds and description == "abortthisnow.") or readability > 30:
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
                    textstat.text_standard(text, float_output=True) if text else -1
                )
            except:
                readability = -1

    new_title = guess_form_name(text, creds=openai_creds) if openai_creds else ""
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
            matches = re.search("(.*)\n", text)
            if matches:
                title = re_case(matches.group(1).strip())
            else:
                title = "(Untitled)"
    nsmi = spot(title + ". " + text, token=spot_token) if spot_token else []
    if normalize:
        length = len(field_names)
        last = "null"
        new_names = []
        new_names_conf = []
        for i, field_name in enumerate(field_names):
            new_name, new_confidence = normalize_name(
                jur or "", cat or "", i, i / length, last, field_name
            )
            new_names.append(new_name)
            new_names_conf.append(new_confidence)
            last = field_name
        new_names = [
            v + "__" + str(new_names[:i].count(v) + 1) if new_names.count(v) > 1 else v
            for i, v in enumerate(new_names)
        ]
    else:
        new_names = field_names
        new_names_conf = []

    tokenized_sentences = sent_tokenize(original_text)
    # No need to detect passive voice in very short sentences
    sentences = [s for s in tokenized_sentences if len(s.split(" ")) > 2]

    try:
        passive_sentences = get_passive_sentences(sentences)
        passive_sentences_count = len(passive_sentences)
    except ValueError:
        passive_sentences_count = 0
        passive_sentences = []

    citations = get_citations(original_text, tokenized_sentences)
    word_count = len(text.split(" "))
    all_caps_count = all_caps_words(text)
    field_types = field_types_and_sizes(ff)
    classified = [
        classify_field(field, new_names[index])
        for index, field in enumerate(field_types)
    ]
    slotin_count = sum(1 for c in classified if c == AnswerType.SLOT_IN)
    gathered_count = sum(1 for c in classified if c == AnswerType.GATHERED)
    third_party_count = sum(1 for c in classified if c == AnswerType.THIRD_PARTY)
    created_count = sum(1 for c in classified if c == AnswerType.CREATED)
    sentence_count = sum(1 for _ in sentences)
    field_count = len(field_names)
    difficult_words = textstat.difficult_words_list(text)
    difficult_word_count = len(difficult_words)
    citation_count = len(citations)
    stats = {
        "title": title,
        "suggested title": new_title,
        "description": description,
        "category": cat,
        "pages": pages_count,
        "reading grade level": readability,
        "time to answer": time_to_answer_form(field_types_and_sizes(ff), new_names)
        if ff
        else [-1, -1],
        "list": nsmi,
        "avg fields per page": f_per_page,
        "fields": new_names,
        "fields_conf": new_names_conf,
        "fields_old": field_names,
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
        "third party percent": third_party_count / field_count
        if field_count > 0
        else 0,
        "passive voice percent": (
            passive_sentences_count / sentence_count if sentence_count > 0 else 0
        ),
        "citations per field": citation_count / field_count if field_count > 0 else 0,
        "citation count": citation_count,
        "all caps percent": all_caps_count / word_count,
        "difficult words": difficult_words,
        "difficult word count": difficult_word_count,
        "difficult word percent": difficult_word_count / word_count,
        "calculation required": needs_calculations(text),
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
                fields_too[k].T = re.sub("^\*", "", field_name)
            my_pdf.save(in_file)
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
        {"name": "sentences per page", "weight": 0.05},
        # percents will have a higher weight, because they are between 0 and 1
        {"name": "slotin percent", "weight": 2},
        {"name": "gathered percent", "weight": 5},
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
            val = min(max(stats[name], metric["clip"][0]), metric["clip"][1])
        elif isinstance(stats[name], bool):
            val = 1 if stats[name] else 0
        else:
            val = stats[name]
        if "intercept" in metric:
            val -= metric["intercept"]
        return val * weight

    return [(m["name"], stats[m["name"]], weight(stats, m)) for m in metrics]


def form_complexity(stats):
    """Gets a single number of how hard the form is to complete. Higher is harder."""
    metrics = _form_complexity_per_metric(stats)
    return sum(val[2] for val in metrics)
