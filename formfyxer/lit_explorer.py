from ctypes.wintypes import SHORT
from dataclasses import Field
import enum
import os
import re
from sklearn.metrics import classification_report
import spacy
from pdfminer.high_level import extract_text
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
from typing import Union, BinaryIO, Iterable, List, Dict, Tuple, Callable, TypedDict
from pathlib import Path

stop_words = set(stopwords.words("english"))

try:
    # this takes a while to load
    import en_core_web_lg

    nlp = en_core_web_lg.load()
except:
    print("Downloading word2vec model en_core_web_lg")
    import subprocess

    bashCommand = "python -m spacy download en_core_web_lg"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    print(f"output of word2vec model download: {str(output)}")
    import en_core_web_lg

    nlp = en_core_web_lg.load()

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
    spot_token = in_file.read().rstrip()


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


def recursive_get_id(values_to_unpack: Union[dict, list], tmpl: set = None):
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
):
    """
    Call the Spot API (https://spot.suffolklitlab.org) to classify the text of a PDF using
    the NSMIv2/LIST taxonomy (https://taxonomy.legal/), but returns only the IDs of issues found in the text.
    """
    headers = {
        "Authorization": "Bearer " + spot_token,
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


def normalize_name(jur: str, group: str, n: int, per, last_field: str, this_field: str):
    """Add hard coded conversions maybe by calling a function
    if returns 0 then fail over to ML or other way around poor prob -> check hard-coded"""

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
    model = AffinityPropagation(damping=damping)
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


def get_existing_pdf_fields(
    in_file: Union[str, Path, BinaryIO, pikepdf.Pdf]
) -> Iterable:
    """
    Use PikePDF to get fields from the PDF
    """
    if isinstance(in_file, pikepdf.Pdf):
        in_pdf = in_file
    else:
        in_pdf = pikepdf.Pdf.open(in_file)
    return [
        {"type": str(field.FT), "var_name": str(field.T), "all": field}
        for field in iter(in_pdf.Root.AcroForm.Fields)
    ]


def get_character_count(
    field: pikepdf.Object, char_width: float = 6, row_height: float = 16
) -> int:
    if not hasattr(field["all"], "Rect"):
        return 1

    # https://pikepdf.readthedocs.io/en/latest/api/main.html#pikepdf.Rectangle
    # Rectangle with llx,lly,urx,ury
    height = field["all"].Rect[3] - field["all"].Rect[1]  # type: ignore
    width = field["all"].Rect[2] - field["all"].Rect[0]  # type: ignore
    # height = field["all"].Rect.height
    # width = field["all"].Rect.width
    num_rows = int(height / row_height) if height > row_height else 1  # type: ignore
    num_cols = int(width / char_width)  # type: ignore

    max_chars = num_rows * num_cols
    return max_chars


class InputType(Enum):
    """
    Input type maps onto the type of input the PDF author chose for the field. We only
    handle text, checkbox, and signature fields.
    """

    TEXT = "text"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"


class FieldInfo(TypedDict):
    var_name: str
    max_length: int
    type: Union[InputType, str]


def field_types_and_sizes(
    fields: Iterable,
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
    for field in fields:
        item: FieldInfo = {
            "var_name": field["var_name"],
            "max_length": get_character_count(
                field,
            ),
            "type": "",
        }
        if field["type"] == "/Tx":
            item["type"] = InputType.TEXT
        elif field["type"] == "/Btn":
            item["type"] = InputType.CHECKBOX
        elif field["type"] == "/Sig":
            item["type"] = InputType.SIGNATURE
        else:
            item["type"] = str(field["type"])
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

    SLOT_IN = "slot in"
    GATHERED = "gathered"
    THIRD_PARTY = "third party"
    CREATED = "created"
    AFFIDAVIT = "affidavit"


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
) -> Callable[[], float]:
    """
    Apply a heuristic for the time it takes to answer the given field, in minutes.
    It is hand-written for now.

    It will factor in the input type, the answer type (slot in, gathered, third party or created), and the
    amount of input text allowed in the field.

    The return value is a tuple of our estimate and a constructed standard deviation
    """
    # Average CPM is about 40: https://en.wikipedia.org/wiki/Words_per_minute#Handwriting
    # Standard deviation is about 17 characters/minute

    # Add mean amount of time for gathering or creating the answer itself (if any) + standard deviation
    TIME_TO_MAKE_ANSWER = {
        AnswerType.SLOT_IN: (0.25, 0.1),
        AnswerType.GATHERED: (3, 2),
        AnswerType.THIRD_PARTY: (5, 2),
        AnswerType.CREATED: (5, 4),
        AnswerType.AFFIDAVIT: (5, 4),
    }

    kind = classify_field(field, new_name)

    if field["type"] == InputType.SIGNATURE or "signature" in field["var_name"]:
        return lambda: np.random.normal(loc=0.5, scale=0.1)
    if field["type"] == InputType.CHECKBOX:
        return lambda: np.random.normal(
            loc=TIME_TO_MAKE_ANSWER[kind][0], scale=TIME_TO_MAKE_ANSWER[kind][1]
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

        return lambda: np.random.normal(
            loc=time_to_write_answer, scale=time_to_write_std_dev
        ) + np.random.normal(
            loc=TIME_TO_MAKE_ANSWER[kind][0], scale=TIME_TO_MAKE_ANSWER[kind][1]
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

    times_to_answer: List[Callable] = []

    for index, field in enumerate(processed_fields):
        times_to_answer.append(time_to_answer_field(field, normalized_fields[index]))

    # Run a monte carlo simulation to get a time to answer and standard deviation
    samples = []
    for _ in range(0, 20000):
        samples.append(sum([item() for item in times_to_answer]))

    np_array = np.array(samples)
    return sigfig.round(np_array.mean(), 2), sigfig.round(np_array.std(), 2)


def unlock_pdf_in_place(in_file: str):
    """
    Try using pikePDF to unlock the PDF it it is locked. This won't work if it has a non-zero length password.
    """
    if not isinstance(in_file, str):
        return
    pdf_file = pikepdf.open(in_file, allow_overwriting_input=True)
    if pdf_file.is_encrypted:
        pdf_file.save(in_file)


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


def parse_form(
    in_file: str,
    title: str = None,
    jur: str = None,
    cat: str = None,
    normalize: bool = True,
    use_spot: bool = False,
    rewrite: bool = False,
    debug: bool = False,
):
    """
    Read in a pdf, pull out basic stats, attempt to normalize its form fields, and re-write the in_file with the new fields (if `rewrite=1`).
    """
    unlock_pdf_in_place(in_file)
    f = pikepdf.open(in_file)

    npages = len(f.pages)

    try:
        with time_limit(15):
            ff = get_existing_pdf_fields(f)
    except TimeoutException as e:
        print("Timed out!")
        ff = None
    except AttributeError:
        ff = None

    if ff:
        fields = [field["var_name"] for field in ff]
    else:
        fields = []
    f_per_page = len(fields) / npages
    original_text = extract_text(in_file)
    text = cleanup_text(original_text)

    if title is None:
        matches = re.search("(.*)\n", text)
        if matches:
            title = re_case(matches.group(1).strip())
        else:
            title = "(Untitled)"

    try:
        if text != "":
            readability = textstat.text_standard(text, float_output=True)
        else:
            readability = -1
    except:
        readability = -1

    if use_spot:
        nmsi = spot(title + ". " + text)
    else:
        nmsi = []

    if normalize:
        i = 0
        length = len(fields)
        last = "null"
        new_fields = []
        new_fields_conf = []
        for field in fields:
            # print(jur,cat,i,i/length,last,field)
            this_field, this_conf = normalize_name(
                jur or "", cat or "", i, i / length, last, field
            )
            new_fields.append(this_field)
            new_fields_conf.append(this_conf)
            last = field

        new_fields = [
            v + "__" + str(new_fields[:i].count(v) + 1)
            if new_fields.count(v) > 1
            else v
            for i, v in enumerate(new_fields)
        ]
    else:
        new_fields = fields
        new_fields_conf = []

    sentences = sent_tokenize(text)

    # Sepehri, A., Markowitz, D. M., & Mir, M. (2022, February 3). PassivePy: A Tool to Automatically Identify Passive Voice in Big Text Data. Retrieved from psyarxiv.com/bwp3t
    passive_text_df = passivepy.match_corpus_level(pd.DataFrame(sentences), 0)

    passive_sentences = len(passive_text_df[passive_text_df["binary"] > 0])

    citations = eyecite.get_citations(
        eyecite.clean_text(original_text, ["all_whitespace", "underscores"])
    )

    stats = {
        "title": title,
        "category": cat,
        "pages": npages,
        "reading grade level": readability,
        "time to answer": time_to_answer_form(field_types_and_sizes(ff), new_fields)
        if ff
        else -1,
        "list": nmsi,
        "avg fields per page": f_per_page,
        "fields": new_fields,
        "fields_conf": new_fields_conf,
        "fields_old": fields,
        "text": text,
        "original_text": original_text,
        "number of sentences": len(sentences),
        "number of passive voice sentences": passive_sentences,
        "number of all caps words": all_caps_words(text),
        "citations": [cite.matched_text() for cite in citations],
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
                        classify_field(field, new_fields[index])
                    ),
                    "time to answer": time_to_answer_field(field, new_fields[index])(),
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

            for k, field in enumerate(new_fields):
                # print(k,field)
                fields_too[k].T = re.sub("^\*", "", field)

            my_pdf.save(in_file)
        except:
            error = "could not change form fields"

    return stats


def form_complexity(text, fields, reading_lv):

    # check for fields that require user to look up info, when found add to complexity
    # maybe score these by minutes to recall/fill out
    # so, figure out words per minute, mix in with readability and page number and field numbers

    return 0
