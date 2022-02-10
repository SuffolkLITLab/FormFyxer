# FormFyxer
A Python package with a collection of functions for learning about and pre-processing pdf forms and associated form fields. This processing is done with an eye towards interoperability with the Suffolk LIT Lab's [Document Assembly Line Project](https://suffolklitlab.org/docassemble-AssemblyLine-documentation/).

## Installation and updating
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install FormFyxer.
Rerun this command to check for and install updates.
```bash
pip install git+https://github.com/SuffolkLITLab/FormFyxer
```

## Functions

- [reCase](#formfyxerrecasetext)
- [regex_norm_field](#formfyxerregex_norm_fieldtext)
- [reformat_field](#formfyxerreformat_fieldtextmax_length30)
- [normalize_name](#formfyxernormalize_namejurgroupnperlast_fieldthis_field)
- [spot](#formfyxerspottextlower025pred05upper06verbose0)
- [parse_form](#formfyxerparse_formfileloctitlenonejurnonecatnonenormalize1use_spot0rewrite0)
- [cluster_screens](#formfyxercluster_screensfieldsdamping07)



### formfyxer.reCase(text)
Reformats snake_case, camelCase, and similarly-formated text into individual words.
#### Parameters:
* **text : str**
#### Returns: 
A string where words combined by cases like snake_case are split back into individual words. 
#### Example:
```python
>>> import formfyxer
>>> formfyxer.reCase("Reformat snake_case, camelCase, and similarly-formated text into individual words.")
'Reformat snake case, camel Case, and similarly formated text into individual words.'
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



### formfyxer.spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0)
A simple wrapper for the LIT Lab's NLP issue spotter [Spot](https://app.swaggerhub.com/apis-docs/suffolklitlab/spot/). In order to use this feature **you must edit the spot_token.txt file found in this package to contain your API token**. You can sign up for an account and get your token on the [Spot website](https://spot.suffolklitlab.org/).

Given a string, this function will return a list of LIST entities/issues found in the text. Items are filtered by estimates of how likely they are to be present. The values dictating this filtering are controlled by the optional `lower`, `pred`, and `upper` paremeters. These refer to the lower bound of the predicted likelihood that an entity is present, the predicted likelihood it is present, and the upper-bound of this prediction respectively. 

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
* **damping : float** A number betwen 0.5 and 1 controlling how similar members of a group need to be. 
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
        "plantiffs1_name",
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
 'screen_1': ['plantiffs1_name',
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
