# Updated on 2022-02-11

import os
import re
import spacy
import PyPDF2
import pikepdf
import textstat
import requests
import json
import numpy as np
from numpy import unique
from numpy import where
from sklearn.cluster import AffinityPropagation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from joblib import load
import nltk
try:
    from nltk.corpus import stopwords
    stopwords.words
except:
    print("Downloading stopwords")
    nltk.download('stopwords')
    from nltk.corpus import stopwords
import math
import signal
from contextlib import contextmanager
import threading
import _thread
import subprocess

stop_words = set(stopwords.words('english'))

try:
    nlp = spacy.load('en_core_web_lg') # this takes a while to loadimport os
except:
    print("Downloading word2vec model en_core_web_lg")
    bashCommand = "python -m spacy download en_core_web_lg"
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    nlp = spacy.load('en_core_web_lg') # this takes a while to loadimport os

# Load local variables, models, and API key(s).

included_fields = load(os.path.join(os.path.dirname(__file__), 'data', 'included_fields.joblib'))
jurisdictions = load(os.path.join(os.path.dirname(__file__), 'data', 'jurisdictions.joblib'))
groups = load(os.path.join(os.path.dirname(__file__), 'data', 'groups.joblib'))
clf_field_names = load(os.path.join(os.path.dirname(__file__), 'data', 'clf_field_names.joblib'))
with open(os.path.join(os.path.dirname(__file__), 'keys', 'spot_token.txt'), 'r') as file:
    spot_token = file.read().rstrip()

# This creates a timeout exception that can be triggered when something hangs too long. 

class TimeoutException(Exception): pass
@contextmanager
def time_limit(seconds):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out.")
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()
#    def signal_handler(signum, frame):
#        raise TimeoutException("Timed out!")
#    signal.signal(signal.SIGALRM, signal_handler)
#    signal.alarm(seconds)
#    try:
#        yield
#    finally:
#        signal.alarm(0)

# Pull ID values out of the LIST/NSMI results from Spot.

def recursive_get_id(values_to_unpack, tmpl=None):
    # h/t to Quinten and Bryce for this code ;)
    if not tmpl:
        tmpl = set()
    if isinstance(values_to_unpack, dict):
        tmpl.add(values_to_unpack.get('id'))
        if values_to_unpack.get('children'):
            tmpl.update(recursive_get_id(values_to_unpack.get('children'), tmpl))
        return tmpl
    elif isinstance(values_to_unpack, list):
        for item in values_to_unpack:
            tmpl.update(recursive_get_id(item, tmpl))
        return tmpl
    else:
        return set()

# Call the Spot API, but return only the IDs of issues found in the text.

def spot(text,lower=0.25,pred=0.5,upper=0.6,verbose=0):

    headers = { "Authorization": "Bearer " + spot_token, "Content-Type":"application/json" }

    body = {
      "text": text,
      "save-text": 0,
      "cutoff-lower": lower,
      "cutoff-pred": pred,
      "cutoff-upper": upper
    }

    r = requests.post('https://spot.suffolklitlab.org/v0/entities-nested/', headers=headers, data=json.dumps(body))
    output_ = r.json()
    
    try:
        output_["build"]
        if verbose!=1:
            try:
                return list(recursive_get_id(output_["labels"]))
            except:
                return []
        else:
            return output_
    except:
        return output_

# A function to pull words out of snake_case, camelCase and the like.

def reCase(text):
    output = re.sub("(\w|\d)(_|-)(\w|\d)","\\1 \\3",text.strip())
    output = re.sub("([a-z])([A-Z]|\d)","\\1 \\2",output)
    output = re.sub("(\d)([A-Z]|[a-z])","\\1 \\2",output)
    output = re.sub("([A-Z]|[a-z])(\d)","\\1 \\2",output)
    return output

# Takes text from an auto-generated field name and uses regex to convert it into an Assembly Line standard field.
# See https://suffolklitlab.org/docassemble-AssemblyLine-documentation/docs/label_variables/

def regex_norm_field(text):
    
    regex_list = [

        # Personal info
        ## Name & Bio
        ["^((My|Your|Full( legal)?) )?Name$","users1_name"],
        ["^(Typed or )?Printed Name\s?\d*$","users1_name"],
        ["^(DOB|Date of Birth|Birthday)$","users1_birthdate"],
        ## Address
        ["^(Street )?Address$","users1_address_line_one"],
        ["^City State Zip$","users1_address_line_two"],
        ["^City$","users1_address_city"],
        ["^State$","users1_address_state"],
        ["^Zip( Code)?$","users1_address_zip"],
        ## Contact
        ["^(Phone|Telephone)$","users1_phone_number"],
        ["^Email( Adress)$","users1_email"],

        # Parties
        ["^plaintiff\(?s?\)?$","plantiff1_name"],
        ["^defendant\(?s?\)?$","defendant1_name"],
        ["^petitioner\(?s?\)?$","petitioners1_name"],
        ["^respondent\(?s?\)?$","respondents1_name"],

        # Court info
        ["^(Court\s)?Case\s?(No|Number)?\s?A?$","docket_number"],
        ["^File\s?(No|Number)?\s?A?$","docket_number"],

        # Form info
        ["^(Signature|Sign( here)?)\s?\d*$","users1_signature"],
        ["^Date\s?\d*$","signature_date"],
    ]

    for regex in regex_list:
        text = re.sub(regex[0],regex[1],text, flags=re.IGNORECASE)
    return text

# Tranforms a string of text into a snake_case variable close in length to `max_length` name by summarizing the string and stiching the summary together in snake_case. h/t h/t https://towardsdatascience.com/nlp-building-a-summariser-68e0c19e3a93

def reformat_field(text,max_length=30):
    orig_title = text.lower()
    orig_title = re.sub("[^a-zA-Z]+"," ",orig_title)
    orig_title_words = orig_title.split()

    deduped_sentence = []
    for word in orig_title_words:
        if word not in deduped_sentence:
            deduped_sentence.append(word)

    filtered_sentence = [w for w in deduped_sentence if not w.lower() in stop_words]

    filtered_title_words = filtered_sentence

    characters = len(' '.join(filtered_title_words))
    
    if characters > 0:

        words = len(filtered_title_words)
        av_word_len = math.ceil(len(' '.join(filtered_title_words))/len(filtered_title_words))
        x_words = math.floor((max_length)/av_word_len)

        sim_mat = np.zeros([len(filtered_title_words),len(filtered_title_words)])
        # for each word compared to other
        for i in range(len(filtered_title_words)):
            for j in range(len(filtered_title_words)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity(nlp(filtered_title_words[i]).vector.reshape(1,300), nlp(filtered_title_words[j]).vector.reshape(1,300))[0,0]

        try:
            nx_graph = nx.from_numpy_array(sim_mat)
            scores = nx.pagerank(nx_graph)
            sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)

            if x_words > len(scores):
                x_words=len(scores)

            i = 0
            new_title = ""
            for x in filtered_title_words:
                if scores[i] >= sorted_scores[x_words-1][1]:
                    if len(new_title)>0: new_title+="_"
                    new_title += x
                i+=1

            return new_title
        except:
            return '_'.join(filtered_title_words)
    else:
        if re.search("^(\d+)$", text):
            return "unknown"
        else:
            return re.sub("\s+","_",text.lower())

# Normalize a word vector.

def norm(row):
    try:
        matrix = row.reshape(1,-1).astype(np.float64)
        return normalize(matrix, axis=1, norm='l1')[0]
    except Exception as e:
        print("===================")
        print("Error: ",e)
        print("===================")
        return np.NaN

# Vectorize a string of text. 

def vectorize(text, normalize=1):
    output = nlp(str(text)).vector
    if normalize==1:
        return norm(output)
    else:
        return output

# Given an auto-generated field name and context from the form where it appeared, this function attempts to normalize the field name. Here's what's going on:
# 1. It will `reCase` the variable text
# 2. Then it will run the output through `regex_norm_field`
# 3. If it doesn't find anything, it will use the ML model `clf_field_names`
# 4. If the prediction isn't very confident, it will run it through `reformat_field`  

def normalize_name(jur,group,n,per,last_field,this_field):

    # Add hard coded conversions maybe by calling a function
    # if returns 0 then fail over to ML or otherway around poor prob -> check hard-coded

    if this_field not in included_fields:
        this_field = reCase(this_field)

        out_put = regex_norm_field(this_field)
        conf = 1.0

        if out_put==this_field:
            params = []
            for item in jurisdictions:
                if jur== item:
                    params.append(1)
                else:
                    params.append(0)
            for item in groups:
                if group== item:
                    params.append(1)
                else:
                    params.append(0)
            params.append(n)
            params.append(per)
            for vec in vectorize(this_field):
                params.append(vec)

            for item in included_fields:
                if last_field==item:
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
            return "*"+out_put,conf # this * is a hack to show when something is in the list of known fields later. I need to fix this
        else:
            return reformat_field(this_field),conf
    else:
        return reformat_field(this_field),conf

# Take a list of AL variables and spits out suggested groupings. Here's what's going on:
# 
# 1. It reads in a list of fields (e.g., `["user_name","user_address"]`)
# 2. Splits each field into words (e.g., turning `user_name` into `user name`)
# 3. It then turns these ngrams/"sentences" into vectors using word2vec. 
# 4. For the collection of fields, it finds clusters of these "sentences" within the semantic space defined by word2vec. Currently it uses Affinity Propagation. See https://machinelearningmastery.com/clustering-algorithms-with-python/

def cluster_screens(fields=[],damping=0.7):
    # Takes in a list (fields) and returns a suggested screen grouping
    # Set damping to value >= 0.5 or < 1 to tune how related screens should be

    vec_mat = np.zeros([len(fields),300])
    for i in range(len(fields)):
        vec_mat[i] = [nlp(reCase(fields[i])).vector][0]

    # create model
    model = AffinityPropagation(damping=damping)
    #model = AffinityPropagation(damping=damping,random_state=4) consider using this to get consitent results. note will have to requier newer version
    # fit the model
    model.fit(vec_mat)
    # assign a cluster to each example
    yhat = model.predict(vec_mat)
    # retrieve unique clusters
    clusters = unique(yhat)

    screens = {}
    #sim = np.zeros([5,300])
    i=0
    for cluster in clusters:
        this_screen = where(yhat == cluster)[0]
        vars = []
        j=0
        for screen in this_screen:
            #sim[screen]=vec_mat[screen] # use this spot to add up vectors for compare to list
            vars.append(fields[screen])
            j+=1
        screens["screen_%s"%i]=vars
        i+=1

    return screens

# Get the text content of a pdf.

def read_pdf (file):
    try:
        pdfFile = PyPDF2.PdfFileReader(open(file, "rb"))
        if pdfFile.isEncrypted:
            try:
                pdfFile.decrypt('')
                #print ('File Decrypted (PyPDF2)')
            except:
                #
                #
                # This didn't go so well on my Windows box so I just ran this in the pdf folder's cmd:
                # for %f in (*.*) do copy %f temp.pdf /Y && "C:\Program Files (x86)\qpdf-8.0.2\bin\qpdf.exe" --password="" --decrypt temp.pdf %f
                #
                #
                #
                
                command="cp "+file+" tmp/temp.pdf; qpdf --password='' --decrypt tmp/temp.pdf "+file
                os.system(command)
                #print ('File Decrypted (qpdf)')
                #re-open the decrypted file
                pdfFile = PyPDF2.PdfFileReader(open(file, "rb"))
        text = ""
        for page in pdfFile.pages:
            text = text + " " + page.extractText()
        text = reCase(text)
        text = re.sub("(\.|,|;|:|!|\?|\n|\]|\))","\\1 ",text)
        text = re.sub("(\(|\[)"," \\1",text)
        text = re.sub(" +"," ",text)
        return text
    except:
        return ""

# Read in a pdf, pull out basic stats, attempt to normalize its form fields, and re-write the file with the new fields (if `rewrite=1`). 

def parse_form(fileloc,title=None,jur=None,cat=None,normalize=1,use_spot=0,rewrite=0):
    f = PyPDF2.PdfFileReader(fileloc)

    if f.isEncrypted:
        pdf = pikepdf.open(fileloc, allow_overwriting_input=True)
        pdf.save(fileloc)
        f = PyPDF2.PdfFileReader(fileloc)
        
    npages = f.getNumPages()
  
    # When reading some pdfs, this can hang due to their crazy field structure
    try:
        with time_limit(15):
            ff = f.getFields()
    except TimeoutException as e:
        print("Timed out!")
        ff = None   
    
    if ff:
        fields = list(ff.keys())
    else:
        fields = []
    f_per_page = len(fields)/npages
    text = read_pdf(fileloc)
    
    if title is None:
        matches = re.search("(.*)\n",text)
        if matches:
            title = reCase(matches.group(1).strip())
        else:
            title = "(Untitled)"        

    try:
        #readbility = int(Readability(text).flesch_kincaid().grade_level)
        text = re.sub("_"," ",text)
        text = re.sub("\n",". ",text)
        text = re.sub(" +"," ",text)
        if text!= "":
            consensus = textstat.text_standard(text)
            readbility = eval(re.sub("^(\d+)[^0-9]+(\d+)\w*.*","(\\1+\\2)/2",consensus))
        else:
            readbility = None
    except:
        readbility = None

    if use_spot==1:
        nmsi = spot(title + ". " +text)      
    else:
        nmsi = []
        
    if normalize==1:
        i = 0 
        length = len(fields)
        last = "null"
        new_fields = []
        new_fields_conf = []
        for field in fields:
            #print(jur,cat,i,i/length,last,field)
            this_field,this_conf = normalize_name(jur,cat,i,i/length,last,field)
            new_fields.append(this_field)
            new_fields_conf.append(this_conf)
            last = field
        
        new_fields = [v + "__" + str(new_fields[:i].count(v) + 1) if new_fields.count(v) > 1 else v for i, v in enumerate(new_fields)]
    else:
        new_fields = fields
        new_fields_conf = []
    
    stats = {
            "title":title,
            "category":cat,
            "pages":npages,
            "reading grade level": readbility,
            "list":nmsi,
            "avg fields per page": f_per_page,
            "fields":new_fields,
            "fields_conf":new_fields_conf,
            "fields_old":fields,
            "text":text
            }    
    
    if rewrite==1:
        try:
            if 1==1:
                my_pdf = pikepdf.Pdf.open(fileloc, allow_overwriting_input=True)
                fields_too = my_pdf.Root.AcroForm.Fields #[0]["/Kids"][0]["/Kids"][0]["/Kids"][0]["/Kids"]
                #print(repr(fields_too))
                
                k =0
                for field in new_fields:
                    #print(k,field)
                    fields_too[k].T = re.sub("^\*","",field)
                    k+=1

                #f2.T = 'new_hospital_name'
                #filename = re.search("\/(\w*\.pdf)$",fileloc).groups()[0]
                #my_pdf.save('/%s'%(filename))
                my_pdf.save(fileloc)
            else:
                file = PdfFileWriter()

                first_page = f.getPage(0)

                file.cloneDocumentFromReader(f)
                #file.appendPagesFromReader(f)

                x ={}
                for y in ff:
                    x[y]=""

                #print(x)

                file.updatePageFormFieldValues(first_page,x)

                output = open('blankPdf.pdf', 'wb')
                file.write(output)  
        except:
            error = "could not change form fields"
    
    return stats

def form_complexity(text,fields,reading_lv):
    
    # check for fields that requier user to look up info, when found add to complexity
    # maybe score these by minutes to recall/fill out
    # so, figure out words per minute, mix in with readability and page number and field numbers
    
    return 0

