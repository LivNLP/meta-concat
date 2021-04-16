"""
Process amazon product reviews and create a corpus.
"""

import json

import nltk
import nltk.data

from nltk.tokenize.stanford import StanfordTokenizer

#nltk.download('all')

sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')


with open("dataset_en_train.json") as F:
    for line in F:
        E = json.loads(line.strip())
        txt = E['review_body'].strip()
        #tok_res = " ".join(sent_detector.tokenize(txt))
        #tok_res = " ".join(StanfordTokenizer().tokenize(txt))
        tok_res = " ".join(nltk.word_tokenize(txt)).lower()
        print(tok_res)