import sys
import unicodedata
from nltk.tokenize.treebank import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

def processed_text(text):
    stripped = strip_accents(text.lower())
    toks = tokenizer.tokenize(stripped)
    return " ".join(toks)

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def clean_uri(uri):
    if uri.startswith("<") and uri.endswith(">"):
        return clean_uri(uri[1:-1])
    elif uri.startswith("\"") and uri.endswith("\""):
        return clean_uri(uri[1:-1])
    return uri

# python process.py test test-peng.txt test.txt
dataset = sys.argv[1]
outfile = open(sys.argv[3], 'w')
with open(sys.argv[2], 'r') as f:
    for i, line in enumerate(f):
        items = line.split(" #### ")
        question = items[0].strip()
        query = strip_accents(items[1].strip().lower())
        if query == "":
            query = strip_accents(question.lower())
        outfile.write("{}-{} %%%% {}\n".format(dataset, i+1, query))

outfile.close()

