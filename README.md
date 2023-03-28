# FormFyxer

[![PyPI version](https://badge.fury.io/py/formfyxer.svg)](https://badge.fury.io/py/formfyxer)

A Python package with a collection of functions for learning about and pre-processing pdf forms and associated form fields. This processing is done with an eye towards interoperability with the Suffolk LIT Lab's [Document Assembly Line Project](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/).

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
TOOLS_TOKEN=<your_token_here> ISUNITTEST=True python -m unittest formfyxer.tests.cluster_test
```

You should test with and without `TOOLS_TOKEN`, and make sure that both pass.

## Functions

Functions from `pdf_wrangling` are found on [our documentation site](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/reference/formfyxer/pdf_wrangling).

- [FormFyxer](#formfyxer)
  - [Installation and updating](#installation-and-updating)
  - [Testing](#testing)
  - [Functions](#functions)
    - [formfyxer.re\_case(text)](#formfyxerre_casetext)
      - [Parameters:](#parameters)
      - [Returns:](#returns)
      - [Example:](#example)
    - [formfyxer.regex\_norm\_field(text)](#formfyxerregex_norm_fieldtext)
      - [Parameters:](#parameters-1)
      - [Returns:](#returns-1)
      - [Example:](#example-1)
    - [formfyxer.reformat\_field(text,max\_length=30)](#formfyxerreformat_fieldtextmax_length30)
      - [Parameters:](#parameters-2)
      - [Returns:](#returns-2)
      - [Example:](#example-2)
    - [formfyxer.normalize\_name(jur,group,n,per,last\_field,this\_field)](#formfyxernormalize_namejurgroupnperlast_fieldthis_field)
      - [Parameters:](#parameters-3)
      - [Returns:](#returns-3)
      - [Example:](#example-3)
    - [formfyxer.vectorize(text,normalize=0)](#formfyxervectorizetextnormalize0)
      - [Parameters:](#parameters-4)
      - [Returns:](#returns-4)
      - [Example:](#example-4)
    - [formfyxer.spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0)](#formfyxerspottextlower025pred05upper06verbose0)
      - [Parameters:](#parameters-5)
      - [Returns:](#returns-5)
      - [Example:](#example-5)
    - [formfyxer.guess\_form\_name(text)](#formfyxerguess_form_nametext)
      - [Parameters:](#parameters-6)
      - [Returns:](#returns-6)
      - [Example:](#example-6)
    - [formfyxer.plain\_lang(text)](#formfyxerplain_langtext)
      - [Parameters:](#parameters-7)
      - [Returns:](#returns-7)
      - [Example:](#example-7)
    - [formfyxer.describe\_form(text)](#formfyxerdescribe_formtext)
      - [Parameters:](#parameters-8)
      - [Returns:](#returns-8)
      - [Example:](#example-8)
    - [formfyxer.parse\_form(fileloc,title=None,jur=None,cat=None,normalize=1,use\_spot=0,rewrite=0)](#formfyxerparse_formfileloctitlenonejurnonecatnonenormalize1use_spot0rewrite0)
      - [Parameters:](#parameters-9)
      - [Returns:](#returns-9)
      - [Example:](#example-9)
    - [formfyxer.cluster\_screens(fields,damping=0.7)](#formfyxercluster_screensfieldsdamping07)
      - [Parameters:](#parameters-10)
      - [Returns:](#returns-10)
      - [Example:](#example-10)
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
Given an auto-generated field name (e.g., those applied by a PDF editor's find form feilds function), this function uses regular expressions to replace common auto-generated field names for those found in our [standard field names](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/label_variables/). 
#### Parameters:
* **text : str** A string of words, such as that found in an auto-generated field name (e.g., those applied by a PDF editor's find form feilds function).
#### Returns: 
Either the original string/field name, or if a standard field name is found, the standard field name.
#### Example:
```python
>>> import formfyxer
>>> formfyxer.regex_norm_field("your name")
'users1_name'
```
[back to top](#formfyxer)



### formfyxer.reformat_field(text,max_length=30)
Given a string of words, this function provides a summary of the string's semantic content by boiling it down to a few words. It then reformats these keywords into snake_case. 
#### Parameters:
* **text : str** A string of words.
* **max_length :  int** An integer setting the maximum length of your field name.
#### Returns: 
A snake_case string summarizing the input sentence. 
#### Example:
```python
>>> import formfyxer
>>> reformat_field("this is a variable where you fill out your name")
'variable_fill_name'
```
[back to top](#formfyxer)



### formfyxer.normalize_name(jur,group,n,per,last_field,this_field)
This function will use the above functions to produce a field name conforming to the format of our [standard field names](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/label_variables/). It does this first by applying `reCase()` to the text of a field. It then applies `regex_norm_field()`. If a standard field name is NOT found, it makes use of a machine learning model we have trained to classify the text as one of our standard field names. If the model is confident in a classification, it changes the text to that field name. If it us uncertian, it applies `reformat_field()`. The end result is that you can feed in a field name and receive output that has been converted into either one of our standard fields or a string of similar formatting. 
#### Parameters:
* **jur : str** The two-letter US postal jurisdiction code (e.g., MA).
* **group : str** Eventually this should be a LIST issue, but right now it can be anything. 
* **n : int** The count of what number this field this is on its form (e.g., if it's the first field n=1) 
* **per : float {0-1)** n divided by the total number of fields on this form. That is, the percentage of the form that completion of this field will result in
* **last_field : str** The normalized field name of the field that preceeded this one. 
* **this_field : str** The un-normalized (raw) field name of the field you are looking to normalize. 
#### Returns: 
object 
#### Example:
```python
>>> import formfyxer
>>> normalize_name("UT",None,2,0.3,"null","Case Number")
('*docket_number', 1.0)
```
[back to top](#formfyxer)




### formfyxer.vectorize(text,normalize=0)
A simple wrapper for Spacy's word2vec vectorization of a string. 

#### Parameters:
* **text : str** Text.
* **normalize : 0 or 1, default 1** If set to 1 vector will be normalized. 
#### Returns: 
A 300d vector for the text provided, using word2vec.
#### Example:
```python
>>> import formfyxer
>>> 
>>> formfyxer.vectorize("my landlord kicked me out", normalize=1)
array([-5.26231120e-04,  2.24983986e-03, -8.35795340e-03,  4.02475413e-03,
        3.44079169e-03, -3.62503832e-03,  4.91300346e-04, -1.02481993e-02,
       -8.75018570e-05,  5.77801012e-02, -8.04772768e-03,  1.93668896e-03,
        1.61031034e-03, -4.88554112e-03, -7.56827288e-03, -3.22198853e-04,
        1.72684901e-03,  1.09334913e-02, -2.45365698e-03,  2.60785779e-03,
        3.31795751e-03, -1.82501888e-03, -5.17577020e-04,  9.05366796e-04,
       -1.88947119e-03, -1.41778216e-03, -2.19670966e-03, -2.33783632e-03,
        1.00638480e-03, -6.26632172e-03, -5.01368841e-04,  7.08620072e-03,
       -3.12600359e-03,  6.44426321e-03,  2.27485859e-04,  8.98271860e-04,
       -2.61456956e-03, -6.52393141e-04, -1.24109763e-03, -3.89325497e-03,
       -3.06367232e-03, -1.28471724e-03, -2.69054515e-03, -3.91209299e-03,
        1.32449560e-03,  4.50141250e-03, -1.75082921e-03, -3.78464401e-03,
        1.40550716e-03,  2.89970543e-03, -2.89665523e-03,  2.99134455e-03,
       -4.17377978e-03,  2.69617527e-03, -1.59275456e-04, -7.83068891e-04,
       -6.36623462e-04, -2.48208915e-03, -1.25590225e-03, -2.50187579e-04,
       -2.05267708e-03, -2.68196150e-03, -4.38043172e-05,  4.18123381e-03,
        6.31226077e-03, -3.65403513e-03,  2.65449648e-03,  2.05167042e-04,
       -1.30922426e-03,  8.00734152e-03,  1.38796774e-03,  1.76862839e-03,
        7.66475223e-03,  8.80032085e-04,  2.59798026e-03,  3.22615966e-03,
        9.56395168e-03, -4.76434876e-03, -4.72719918e-03,  6.81382797e-03,
       -4.05296658e-03,  8.91728523e-03, -4.68128831e-03, -5.16060328e-03,
        1.66344554e-03,  2.91576202e-03,  7.22084550e-03, -5.06330498e-03,
        1.33031214e-02,  4.79664028e-03, -1.07973688e-03,  3.88623763e-05,
       -1.23686194e-04,  3.57764797e-04,  2.40055549e-03,  1.39446255e-03,
        7.12260341e-03, -8.41163896e-04, -5.16814871e-04,  4.72550955e-03,
        1.33025948e-03, -4.51154774e-03,  8.67997034e-05, -5.33963124e-03,
       -2.16460094e-03, -1.51862293e-02,  5.61281296e-03,  4.78538005e-03,
       -4.88630835e-03, -2.68976954e-03,  6.91397406e-04, -7.68237538e-03,
        6.35172810e-03,  3.49582726e-04,  2.25623079e-03,  7.04172971e-04,
        3.25000592e-03, -1.08145548e-03,  5.15488738e-03, -2.57177091e-04,
        1.76189272e-03, -1.11836531e-04, -2.16169944e-04, -1.91144984e-03,
        3.43246164e-03,  3.58958300e-03, -5.56801994e-03,  1.46526319e-04,
       -3.65503599e-03,  1.21198749e-03, -4.71577666e-03,  1.32155107e-03,
       -7.55368584e-03,  4.21072481e-04,  3.84473030e-03,  7.37876031e-04,
        2.51003285e-03,  1.13599304e-02, -2.91033290e-03, -2.17953879e-03,
       -4.48384314e-02, -2.55559416e-03,  1.27762248e-03,  3.16679245e-04,
        1.73199350e-04,  1.50888037e-03, -4.79663957e-03,  5.17604744e-03,
       -3.98579436e-03, -1.39384994e-03, -1.60889822e-03,  6.14265860e-03,
        2.33297705e-04, -3.52656403e-04,  1.12852005e-03,  1.65170200e-03,
       -1.63151391e-03, -6.32309832e-04,  1.19076047e-03, -3.06317795e-03,
       -4.79622367e-03, -3.06260930e-04,  1.65514420e-03, -1.22830785e-03,
       -3.04184669e-03, -4.55402836e-03,  3.88777805e-03,  2.00426727e-03,
        2.14710762e-03,  1.68567815e-03, -6.47409360e-03, -6.46973133e-03,
       -5.74770121e-05, -5.43416808e-03, -6.19798278e-03, -1.14207302e-04,
       -1.01290520e-03,  9.50722038e-04, -4.10705805e-03,  4.52319444e-04,
        4.62367242e-03, -5.16681711e-03, -3.19836917e-03,  2.15407870e-03,
        2.97252097e-03,  1.56973282e-03, -3.55476436e-03,  1.62631717e-03,
        2.02915202e-03,  2.61959652e-03, -3.41814925e-04,  1.05913682e-03,
       -5.67106089e-03, -2.26600230e-04,  4.77448347e-04,  6.28645666e-03,
       -1.01146419e-03, -6.35369059e-03,  1.65369797e-04, -9.55562091e-04,
       -5.87664628e-03,  1.26514872e-03, -7.70650378e-03,  1.81958423e-03,
        1.19041591e-03,  3.69227689e-03,  6.81289417e-03,  2.34441887e-04,
        4.76372670e-03, -3.22644252e-03,  7.13130209e-04, -1.34127493e-03,
        3.49982077e-03,  3.36139797e-03, -2.95230863e-03,  5.77470678e-03,
       -1.91699363e-03, -1.58495209e-03, -1.37739101e-03,  2.91823707e-03,
       -2.54773333e-03,  1.70074190e-03, -9.05261306e-04, -1.14740142e-03,
        9.50051913e-04, -4.24206736e-03,  2.38089718e-04,  2.40279602e-03,
       -1.33007275e-03, -5.04937888e-03, -3.60123558e-03,  9.58420218e-04,
        3.30443339e-03,  1.58153860e-03,  4.72644728e-03, -3.60392571e-03,
       -3.44693427e-04, -2.36716859e-03,  4.83650924e-03,  2.43589561e-03,
        3.00392594e-03,  3.94615047e-03,  1.71704478e-03,  5.89554031e-03,
        2.18071977e-03, -6.37284036e-03,  5.64066123e-03, -4.59186528e-03,
       -1.58649190e-03,  6.56146175e-03, -1.04672343e-03, -6.28604576e-03,
       -7.47575222e-04,  2.12077705e-03,  5.26619347e-03,  2.89095824e-03,
       -1.62623299e-03, -5.51077501e-03, -9.67431517e-05,  3.44178292e-03,
       -6.74005028e-03,  3.24305482e-03,  1.03411332e-03, -1.69642029e-03,
        5.81055945e-03, -2.27511233e-03, -1.24854863e-03,  4.16209955e-04,
       -9.86948252e-04, -3.47459984e-03,  8.35964266e-03,  1.90435018e-03,
       -6.13537569e-04, -4.42874018e-03, -2.34987644e-03, -6.47533826e-04,
       -5.74400109e-03, -3.98190719e-03,  1.23454575e-03,  4.60058269e-03,
        1.13744318e-03, -1.43143558e-03, -3.46458480e-03,  1.70765680e-03,
       -5.39483022e-03, -2.10772697e-03,  9.33664766e-03, -2.98427860e-03,
       -6.75659472e-04, -3.30385404e-04, -3.49518099e-03,  1.54804617e-04,
        2.41326119e-03,  4.88547941e-03,  2.49007910e-03, -2.13324702e-04,
        5.94240817e-03,  4.13455749e-03,  1.64867480e-03, -1.49173268e-03])
```

[back to top](#formfyxer)



### formfyxer.spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0)
A simple wrapper for the LIT Lab's NLP issue spotter [Spot](https://app.swaggerhub.com/apis-docs/suffolklitlab/spot/). In order to use this feature **you must edit the spot_token.txt file found in this package to contain your API token**. You can sign up for an account and get your token on the [Spot website](https://spot.suffolklitlab.org/).

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
An OpenAI-enabled tool that will guess the name of a court form given the full text of the form. In order to use this feature **you must edit the `openai_org.txt` and `openai_key.txt` files found in this package to contain your OpenAI credentials**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

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
An OpenAI-enabled tool that will rewrite a text into a plain language draft. In order to use this feature **you must edit the `openai_org.txt` and `openai_key.txt` files found in this package to contain your OpenAI credentials**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

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
An OpenAI-enabled tool that will write a draft plain language description for a form. In order to use this feature **you must edit the `openai_org.txt` and `openai_key.txt` files found in this package to contain your OpenAI credentials**. You can sign up for an account and get your token on the [OpenAI signup](https://beta.openai.com/signup).

Given a string conataining the full text of a court form, this function will return its a draft description of the form written in plain language. 

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



### formfyxer.parse_form(fileloc,title=None,jur=None,cat=None,normalize=1,use_spot=0,rewrite=0)
Read in a pdf with pre-existing form fields, pull out basic stats, attempt to normalize its field names, and re-write the file with the new fields (if rewrite=1). 
#### Parameters:
* **fileloc :** the location of the pdf file to be processed.
* **title : str, default None** The title of the form. If set to `None` the parser will make a best guess. 
* **jur : str, default None** The two-letter US postal jurisdiction code (e.g., MA).
* **cat: list, default None** Eventually this should be a LIST issue, but right now it can be anything. 
* **normalize : {0 or 1}, default 1** O will not attempt to normalize the form fields. 1 will.
* **use_spot : {0 or 1}, default 0** 1 will use spot to guess at LIST issues covered by this form. 0 will skip this.
* **rewrite : {0 or 1}, rewrite 0** 1 will attempt to write over the original file with the normalized fields (assuming normalize set to 1). O will leave the original file as is.
#### Returns: 
Object containing a set of stats for the form. See below
#### Example:
```python
>>> import formfyxer
>>> stats = formfyxer.parse_form("sample.pdf",title=None,jur="UT",cat=None,normalize=0,use_spot=0,rewrite=0)
>>> stats
{'title': 'Waiver of',
 'category': None,
 'pages': 2,
 'reading grade level': 7.5,
 'list': [],
 'avg fields per page': 0.0,
 'fields': [],
 'fields_conf': [],
 'fields_old': [],
 'text': 'Waiver of . Rights. . Approved Board of District Court Judges . December 17, 2010. . Revised . November 26. , 2019. . Page . 1. . of . 2. . . . . Name. . . . . Address. . . . . City, State, Zip. . . . . Phone. . . . Check your email. . You will receive information and . documents at this email address. . . . Email. . In the District Court of Utah. . Judicial District Count. y. . Court Address . . In the Matter of the Adoption of. . . . . (. . ) . . Waiver . of Rights. . . . . . Case Number. . . . . . Jud. ge. . . . . . Commissioner. . Do not sign this document without reading it. . Do not sign it unless everything . stated is true and correct. . If you have any questions, . talk with. . an attorney. . . . . . You have the right to be notified of hearings and . to be served with papers in this . matter. You have the right to intervene and oppose the adoption. . By signing this . document you are giving up . these. . rights. . . . . . If you . want to waive your rights. , complete this form, sign it, . and . return. . it . to the . Petitioner. . . . . . If yo. u . want to intervene and . oppose the adoption, . file a motion to intervene . with . this. . court. . within 30 days after the . Notice of Petition to Adopt. . was served on you. . . 1. . . . I make this statement free from . duress. . . . Waiver of . Rights. . Approved Board of District Court Judges . December 17, 2010. . Revised . November 26. , 2019. . Page . 2. . of . 2. . . 2. . . . I am the . adoptee. . . [ ] . Guardian. . without the right. . to consent to the adoption. . [ ] . Custodian. . [ ] . S. p. ouse. . 3. . . . I understand that. : . . . . I have the right to be notified of hearings and to be served with papers in this . matter. . . . . I have the right to intervene and oppose the adoption. . . . . By signing this document . I am. . givin. g up . these. . rights. . . . 4. . . . Understanding all of this, . I . voluntarily . waive my right to . be notified of hearings . and served with papers in this matter. , and. . I voluntarily waive my right to . intervene in this matter. . . Do not sign this document without reading it. . Do n. ot sign it unless everything . stated is true and correct. . If you have any questions, . talk with. . an attorney. . . . . . I declare under . criminal . penalty . under the law of Utah. . that everything stated . in this document is true. . . . Signed at . (city, and state or country) . . . . Sign. atu. . . . Date. . Printed Name'}
```
[back to top](#formfyxer)



### formfyxer.cluster_screens(fields,damping=0.7)
This function will take a list of snake_case field names and group them by semantic similarity. 
#### Parameters:
* **files : list** A list of snake_case field names.
* **damping : float** A number between 0.5 and 1 controlling how similar members of a group need to be.
#### Returns: 
An object grouping together similar field names.  
#### Example:
```python
>>> import formfyxer
>>> fields= [
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
>>> cluster_screens(fields,damping=0.7)
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


## License
[MIT](https://github.com/SuffolkLITLab/FormFyxer/blob/main/LICENSE)
