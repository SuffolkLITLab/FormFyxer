# FormFyxer

[![PyPI version](https://badge.fury.io/py/formfyxer.svg)](https://badge.fury.io/py/formfyxer)

A Python package with a collection of functions for learning about and pre-processing pdf forms and associated form fields. This processing is done with an eye towards interoperability with the Suffolk LIT Lab's [Document Assembly Line Project](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/).

This repository is the engine for [RateMyPDF](https://ratemypdf.com). It has been described in a paper published in the proceedings
of ICAIL '23. You can view it [here](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/complexity/complexity/#download-and-cite-our-paper).

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FormFyxer.
Rerun this command to check for and install updates directly from GitHub.

```bash
pip install git+https://github.com/SuffolkLITLab/FormFyxer
```

If you are on Mac or Windows, you'll need to install [poppler](https://poppler.freedesktop.org/) for your respective platform.
If you are on Anaconda, simply run `conda install poppler`. Otherwise, follow the instructions here:
- [macOS instructions](https://macappstore.org/poppler/)
- [Windows download](https://github.com/oschwartz10612/poppler-windows/releases/tag/v22.04.0-0)

## Testing

```bash
ISUNITTEST=True python -m unittest formfyxer.tests.cluster_test
```

Set `SPOT_TOKEN` in your environment if you want to exercise Spot-backed features during tests.

## Configuration

Secrets are now loaded from environment variables. Copy `.env.example` to `.env` and fill in any credentials you need:

```bash
cp .env.example .env
```

The library looks for `SPOT_TOKEN` for Spot access and `OPENAI_API_KEY` with an optional `OPENAI_ORGANIZATION` for OpenAI features. Any standard environment variable loader that populates those values will work; the package uses [python-dotenv](https://github.com/theskumar/python-dotenv) to read `.env` automatically.

## Passive Voice Evaluation

You can sanity check the LLM-backed passive voice detector with [Promptfoo](https://promptfoo.dev/).

```bash
promptfoo eval -c promptfooconfig.yaml
```

The eval uses `formfyxer/tests/passive_voice_test_dataset.csv` to ensure sentences labeled as passive produce a non-empty `fragments` array. If you change the prompt inside `formfyxer/passive_voice_detection.py`, copy the updated text (the `system_prompt` and numbering format) into `promptfooconfig.yaml` so the evaluation mirrors runtime behavior.

Raw percentage correct performance using gpt-5 nano on the benchmark dataset is **95.56%.** (Did not
calculate more specific F1 score).

## Functions

Functions from `pdf_wrangling` are found on [our documentation site](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/reference/formfyxer/pdf_wrangling).

- [formfyxer.re_case(text)](#formfyxerre_casetext)
- [formfyxer.regex_norm_field(text)](#formfyxerregex_norm_fieldtext)
- [formfyxer.reformat_field(text,max_length=30,tools_token=None)](#formfyxerreformat_fieldtextmax_length30tools_tokennone)
- [formfyxer.normalize_name(jur,group,n,per,last_field,this_field,tools_token=None)](#formfyxernormalize_namejurgroupnperlast_fieldthis_fieldtools_tokennone)
- [formfyxer.spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0)](#formfyxerspottextlower025pred05upper06verbose0)
- [formfyxer.guess_form_name(text)](#formfyxerguess_form_nametext)
- [formfyxer.plain_lang(text)](#formfyxerplain_langtext)
- [formfyxer.describe_form(text)](#formfyxerdescribe_formtext)
- [formfyxer.parse_form(in_file,title=None,jur=None,cat=None,normalize=True,spot_token=None,tools_token=None,openai_creds=None,openai_api_key=None,rewrite=False,debug=False)](#formfyxerparse_formin_filetitlenonejurnonecatnonenormalizetruespot_tokennonetools_tokennoneopenai_credsnoneopenai_api_keynonerewritefalsedebugfalse)
- [formfyxer.cluster_screens(fields,openai_creds=None,api_key=None,model='gpt-5-nano',damping=None,tools_token=None)](#formfyxercluster_screensfieldsopenai_credsnoneapi_keynonemodelgpt-5-nanodampingnonetools_tokennone)
- [formfyxer.get_sensitive_data_types(fields,fields_old=None)](#formfyxerget_sensitive_data_typesfieldsfields_oldnone)
- [License](#license)

### formfyxer.re_case(text)
Reformats snake_case, camelCase, and similarly-formatted text into individual words.
#### Parameters:
* **text : str**
#### Returns: 
A string where words combined by cases like snake_case are split back into individual words. 
#### Example:
```python
>>> import formfyxer
>>> formfyxer.reCase("Reformat snake_case, camelCase, and similarly-formatted text into individual words.")
'Reformat snake case, camel Case, and similarly formatted text into individual words.'
```
[back to top](#formfyxer)


### formfyxer.regex_norm_field(text)
Uses regular expressions to map common auto-generated field labels to the Suffolk LIT Lab's [standard field names](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/label_variables/).

#### Parameters:
* **text : str** The raw field label to normalize.

#### Returns:
The normalized field name if a pattern matches; otherwise returns the original text.

#### Example:
```python
>>> import formfyxer
>>> formfyxer.regex_norm_field("Signature 1")
'users1_signature'
```
[back to top](#formfyxer)


### formfyxer.reformat_field(text,max_length=30,tools_token=None)
Generates a snake_case label from arbitrary text without relying on external similarity services. The helper filters out stop words, deduplicates repeated tokens, and enforces a configurable maximum length.

#### Parameters:
* **text : str** The field label to reformat.
* **max_length : int, default 30** Maximum number of characters allowed in the generated name.
* **tools_token : str | None, default None** Deprecated. Previously forwarded to tools.suffolklitlab.org and ignored today.

#### Returns:
A snake_case representation of the label that fits within the specified length.

#### Example:
```python
>>> import formfyxer
>>> formfyxer.reformat_field("Your Mailing Address")
'your_mailing_address'
```
[back to top](#formfyxer)


### formfyxer.normalize_name(jur,group,n,per,last_field,this_field,tools_token=None)
Combines `re_case`, `regex_norm_field`, and a lightweight heuristic fallback to normalize PDF field names into Assembly Line conventions. When LLM credentials are provided, the runtime prefers enhanced context-aware normalization; otherwise it uses the traditional pipeline.

#### Parameters:
* **jur : str** Two-letter jurisdiction code (legacy compatibility parameter).
* **group : str** Optional category label (legacy compatibility parameter).
* **n : int** Index of the field within the document.
* **per : float** Percentage through the field list (legacy compatibility parameter).
* **last_field : str** Name of the previously processed field (legacy compatibility parameter).
* **this_field : str** Raw field name to normalize.
* **tools_token : str | None, default None** Deprecated. Accepted for backward compatibility but unused.

#### Returns:
A tuple of `(normalized_name, confidence_score)`.

#### Example:
```python
>>> import formfyxer
>>> formfyxer.normalize_name("UT", None, 2, 0.3, "null", "Case Number")
('*docket_number', 0.5)
```
[back to top](#formfyxer)


### formfyxer.spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0)
A simple wrapper for the LIT Lab's NLP issue spotter [Spot](https://app.swaggerhub.com/apis-docs/suffolklitlab/spot/). To use this feature **configure a `.env` file (see `.env.example`) with your Spot API token in `SPOT_TOKEN`**. You can sign up for an account and get your token on the [Spot website](https://spot.suffolklitlab.org/).

Given a string, this function will return a list of LIST entities/issues found in the text. Items are filtered by estimates of how likely they are to be present. The values dictating this filtering are controlled by the optional `lower`, `pred`, and `upper` parameters. These refer to the lower bound of the predicted likelihood that an entity is present, the predicted likelihood it is present, and the upper-bound of this prediction respectively. 

#### Parameters:
* **text : str** Text describing some fact pattern.
* **lower : float between 0 and 1, default 0.25** Defines the cutoff for the lower bound of a prediction (`lower`) necessary to trigger inclusion in the results. That is, the lower bound of a prediction must exceed `lower` for it to appear in the results.
* **pred : float between 0 and 1, default 0.5** Defines the cutoff for the prediction (`pred`) necessary to trigger inclusion in the results. That is, the prediction must exceed `pred` for it to appear in the results.
* **upper : float between 0 and 1, default 0.6** Defines the cutoff for the upper bound of a prediction (`upper`) necessary to trigger inclusion in the results. That is, the upper bound of a prediction must exceed `upper` for it to appear in the results.
* **verbose : 0 or 1, default 0** If set to 0 will return only a list of LIST IDs. If set to 1, will return a full set of Spot results. 
#### Returns: 
A list of LIST entities/issues found in the text.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.spot("my landlord kicked me out")
['HO-02-00-00-00', 'HO-00-00-00-00', 'HO-05-00-00-00', 'HO-06-00-00-00']

>>> formfyxer.spot("my landlord kicked me out", verbose=1)
{'build': 9,
 'query-id': '1efa5a098bc24f868684339f638ab7eb',
 'text': 'my landlord kicked me out',
 'save-text': 0,
 'cutoff-lower': 0.25,
 'cutoff-pred': 0.5,
 'cutoff-upper': 0.6,
 'labels': [{'id': 'HO-00-00-00-00',
   'name': 'Housing',
   'lower': 0.6614134886446631,
   'pred': 0.7022160833303629,
   'upper': 0.7208275781222152,
   'children': [{'id': 'HO-02-00-00-00',
     'name': 'Eviction from a home',
     'lower': 0.4048013980740931,
     'pred': 0.5571460102525152,
     'upper': 0.6989976788434928},
    {'id': 'HO-05-00-00-00',
     'name': 'Problems with living conditions',
     'lower': 0.3446066253503793,
     'pred': 0.5070074487913626,
     'upper': 0.6326627767849852},
    {'id': 'HO-06-00-00-00',
     'name': 'Renting or leasing a home',
     'lower': 0.6799417713794678,
     'pred': 0.8984004824420323,
     'upper': 0.9210222500232965,
     'children': [{'id': 'HO-02-00-00-00',
       'name': 'Eviction from a home',
       'lower': 0.4048013980740931,
       'pred': 0.5571460102525152,
       'upper': 0.6989976788434928}]}]}]}
```
[back to top](#formfyxer)



### formfyxer.guess_form_name(text)
An OpenAI-enabled tool that will guess the name of a court form given the full text of the form. To use this feature **add your OpenAI credentials to `.env` (see `.env.example`) as `OPENAI_API_KEY` and optionally `OPENAI_ORGANIZATION`**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

Given a string conataining the full text of a court form, this function will return its best guess for the name of the form. 

#### Parameters:
* **text : str** Full text of a form.
#### Returns: 
A string with a proposed name for a court form.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.guess_form_name("""Approved, SCAO. STATE OF MICHIGAN. JUDICIAL CIRCUIT. COUNTY. Original Court. 1st copy Moving party. 2nd copy Objecting party. 3rd copy Friend of the court. 4th copy Proof of service. 5th copy Proof of service. A. CASE NO. OBJECTION TO PROPOSED ORDER. Court address. Court telephone no. Plaintiff's name, address, and telephone no. moving party. Defendant's name, address, and telephone no. moving party. v. Third party's name, address, and telephone no. moving party. I received a notice to enter a proposed order without a hearing dated. I object to the entry of the proposed order and request a hearing by the court. My objection is based on the following reason s. C. B. D. E. Date. Moving party's signature. Name type or print. CERTIFICATE OF MAILING. I certify that on this date I served a copy of this objection on the parties or their attorneys by first class mail addressed to their. last known addresses as defined in MCR 3.203. F. Date. Signature of objecting party. FOC 78 3 11 OBJECTION TO PROPOSED ORDER. MCR 2.602 B.""")
'Objection to Proposed Order'
```
[back to top](#formfyxer)


### formfyxer.plain_lang(text)
An OpenAI-enabled tool that will rewrite a text into a plain language draft. To use this feature **set `OPENAI_API_KEY` (and optionally `OPENAI_ORGANIZATION`) in your `.env` file**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

Given a string, this function will return its attempt at rewriting the srting in plain language. 

#### Parameters:
* **text : str** text.
#### Returns: 
A string with a proposed plain language rewrite.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.guess_form_name("""When the process of freeing a vehicle that has been stuck results in ruts or holes, the operator will fill the rut or hole created by such activity before removing the vehicle from the immediate area.""")
'If you try to free a car that is stuck and it makes a rut or hole, you need to fill it in before you move the car away.'
```
[back to top](#formfyxer)




### formfyxer.describe_form(text)
An OpenAI-enabled tool that will write a draft plain language description for a form. To use this feature **set `OPENAI_API_KEY` (and optionally `OPENAI_ORGANIZATION`) in your `.env` file**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

Given a string containing the full text of a court form, this function will return its a draft description of the form written in plain language. 

#### Parameters:
* **text : str** text.
#### Returns: 
A string with a proposed plain language rewrite.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.guess_form_name("""Approved, SCAO. STATE OF MICHIGAN. JUDICIAL CIRCUIT. COUNTY. Original Court. 1st copy Moving party. 2nd copy Objecting party. 3rd copy Friend of the court. 4th copy Proof of service. 5th copy Proof of service. A. CASE NO. OBJECTION TO PROPOSED ORDER. Court address. Court telephone no. Plaintiff's name, address, and telephone no. moving party. Defendant's name, address, and telephone no. moving party. v. Third party's name, address, and telephone no. moving party. I received a notice to enter a proposed order without a hearing dated. I object to the entry of the proposed order and request a hearing by the court. My objection is based on the following reason s. C. Moving party's signature. Name type or print. CERTIFICATE OF MAILING. Signature of objecting party. I certify that on this date I served a copy of this objection on the parties or their attorneys by first class mail addressed to their. last known addresses as defined in MCR 3.203. FOC 78 3 11 OBJECTION TO PROPOSED ORDER. MCR 2.602 B. B. D. E. Date. F. Date.""")
'This form is used to object to a proposed court order. It allows someone to explain why they disagree with the proposed order and request a hearing by the court.'
```
[back to top](#formfyxer)



### formfyxer.parse_form(in_file,title=None,jur=None,cat=None,normalize=True,spot_token=None,tools_token=None,openai_creds=None,openai_api_key=None,rewrite=False,debug=False)
Read in a PDF, gather statistics, optionally normalize field names, and return a structured summary. When normalization is enabled, the helper can leverage LLM-backed renaming if OpenAI credentials are supplied.

#### Parameters:
* **in_file : str** Path to the PDF to analyze.
* **title : str | None** Explicit title override. When omitted the function guesses a title from metadata, LLM output, or top-of-document text.
* **jur : str | None** Optional jurisdiction hint passed through to `normalize_name` for backward compatibility.
* **cat : str | None** Optional category hint for backward compatibility.
* **normalize : bool, default True** When `True`, attempts to normalize field names using LLM-assisted and heuristic fallbacks.
* **spot_token : str | None** Optional Spot API token enabling NSMI issue detection.
* **tools_token : str | None** Deprecated tools.suffolklitlab.org token. Accepted for older integrations but ignored.
* **openai_creds : dict | None** Mapping with `key`/`org` used for OpenAI calls when provided.
* **openai_api_key : str | None** Explicit API key overriding `openai_creds` and environment variables.
* **rewrite : bool, default False** When `True`, writes normalized fields back into the original PDF.
* **debug : bool, default False** Enables extra logging during processing.

#### Returns:
A dictionary containing metadata, readability scores, Spot results, original and normalized field names, and extracted text snippets.

#### Example:
```python
>>> import formfyxer
>>> stats = formfyxer.parse_form(
...     "sample.pdf",
...     jur="UT",
...     normalize=True,
...     spot_token="YOUR_SPOT_TOKEN",
...     tools_token=None,  # deprecated but still accepted
...     rewrite=False,
... )
>>> stats["pages"], stats["fields"][:3]
(2, ['users1_name', 'users1_address_line_one', 'users1_address_city'])
```
[back to top](#formfyxer)



### formfyxer.cluster_screens(fields,openai_creds=None,api_key=None,model='gpt-5-nano',damping=None,tools_token=None)
Groups snake_case field names into screens using LLM semantic understanding, with a keyword-based fallback when no LLM response is available.

#### Parameters:
* **fields : list[str]** Collection of field names to group.
* **openai_creds : Optional[dict]** Credentials for the OpenAI client (defaults to environment variables).
* **api_key : Optional[str]** Explicit OpenAI API key that overrides credentials and environment variables.
* **model : str** LLM model name (defaults to `gpt-5-nano`).
* **damping : Optional[float]** Deprecated parameter kept for backward compatibility; ignored by the LLM workflow.
* **tools_token : Optional[str]** Deprecated. Previously forwarded to tools.suffolklitlab.org and ignored today.

#### Returns:
A dictionary mapping screen names to lists of field names.

#### Example:
```python
>>> import formfyxer
>>> fields = [
        "users1_name",
        "users1_birthdate",
        "users1_address_line_one",
        "users1_address_line_two",
        "users1_address_city",
        "users1_address_state",
        "users1_address_zip",
        "users1_phone_number",
        "users1_email",
        "plaintiffs1_name",
        "defendants1_name",
        "petitioners1_name",
        "respondents1_name",
        "docket_number",
        "trial_court_county",
        "users1_signature",
        "signature_date"
        ]
>>> cluster_screens(fields)
{'screen_0': ['users1_name',
  'users1_birthdate',
  'users1_address_line_one',
  'users1_address_line_two',
  'users1_address_city',
  'users1_address_state',
  'users1_address_zip',
  'users1_phone_number',
  'users1_email',
  'users1_signature'],
 'screen_1': ['plaintiffs1_name',
  'defendants1_name',
  'petitioners1_name',
  'respondents1_name'],
 'screen_2': ['docket_number'],
 'screen_3': ['trial_court_county'],
 'screen_4': ['signature_date']}
```
[back to top](#formfyxer)



### formfyxer.get_sensitive_data_types(fields, fields_old)
Given a list of fields, identify those related to sensitive information and return a dictionary with the sensitive fields grouped by type. A list of the old field names can also be provided. These fields should be in the same order. Passing the old field names allows the sensitive field algorithm to match more accurately. The return value will not contain the old field name, only the corresponding field name from the first parameter.

The sensitive field types are: Bank Account Number, Credit Card Number, Driver's License Number, and Social Security Number.
#### Parameters:
* **fields : List[str]** List of field names.
#### Returns: 
List of sensitive fields found within the fields passed in.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.get_sensitive_data_types(["users1_name", "users1_address", "users1_ssn", "users1_routing_number"])
{'Social Security Number': ['users1_ssn'], 'Bank Account Number': ['users1_routing_number']}
>>> formfyxer.get_sensitive_data_types(["user_ban1", "user_credit_card_number", "user_cvc", "user_cdl", "user_social_security"], ["old_bank_account_number", "old_credit_card_number", "old_cvc", "old_drivers_license", "old_ssn"])
{'Bank Account Number': ['user_ban1'], 'Credit Card Number': ['user_credit_card_number', 'user_cvc'], "Driver's License Number": ['user_cdl'], 'Social Security Number': ['user_social_security']}
```
[back to top](#formfyxer)



## License
[MIT](https://github.com/SuffolkLITLab/FormFyxer/blob/main/LICENSE)

## Preferred citation format

Please cite this repository as follows:

Quinten Steenhuis, Bryce Willey, and David Colarusso. 2023. Beyond Readability with RateMyPDF: A Combined Rule-based and Machine Learning Approach to Improving Court Forms. In _Proceedings of International Conference on Artificial Intelligence and Law (ICAIL 2023). ACM, New York, NY, USA, 10 pages_. https://doi.org/10.1145/3594536.3595146

Bibtex format:
```bibtex
@article{Steenhuis_Willey_Colarusso_2023, title={Beyond Readability with RateMyPDF: A Combined Rule-based and Machine Learning Approach to Improving Court Forms}, DOI={https://doi.org/10.1145/3594536.3595146}, journal={Proceedings of International Conference on Artificial Intelligence and Law (ICAIL 2023)}, author={Steenhuis, Quinten and Willey, Bryce and Colarusso, David}, year={2023}, pages={287–296}}
```
