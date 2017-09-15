
# coding: utf-8

# In[7]:

import os
import sys
import argparse
import pickle
import math
import unicodedata
import pandas as pd
import numpy as np

from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords


# In[8]:

# arguments
index_entpath = "../indexes/entity_2M.pkl"
index_reachpath = "../indexes/reachability_2M.pkl"
index_namespath = "../indexes/names_2M.pkl"

train_ent_resultpath = "../entity_detection/query-text/train.txt"
train_gold_ent_resultpath = "../entity_detection/gold-query-text/train.txt"
train_rel_resultpath = "../relation_prediction/results/topk-retrieval-train-hits-3.txt"

valid_ent_resultpath = "../entity_detection/query-text/valid.txt"
valid_gold_ent_resultpath = "../entity_detection/gold-query-text/valid.txt"
valid_rel_resultpath = "../relation_prediction/results/topk-retrieval-valid-hits-3.txt"



# In[9]:

tokenizer = TreebankWordTokenizer()
stopwords = set(stopwords.words('english'))

def tokenize_text(text):
    tokens = tokenizer.tokenize(text)
    return tokens

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index

def strip_accents(text):
    return ''.join(c for c in unicodedata.normalize('NFKD', text) if unicodedata.category(c) != 'Mn')


# In[10]:

def get_query_texts(ent_resultpath):
    print("getting query text...")
    lineids = []
    id2query = {}
    notfound = 0
    with open(ent_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            try:
                lineid = items[0].strip()
                queries = items[1:]
                # mid = items[2].strip()
            except:
                # print("ERROR: line does not have >2 items  -->  {}".format(line.strip()))
                notfound += 1
                continue
            # print("{}   -   {}".format(lineid, query))
            lineids.append(lineid)
            id2query[lineid] = queries
    print("notfound (empty query text): {}".format(notfound))
    return lineids, id2query

def get_relations(rel_resultpath):
    print("getting relations...")
    lineids = []
    id2rels = {}
    with open(rel_resultpath, 'r') as f:
        for line in f:
            items = line.strip().split(" %%%% ")
            lineid = items[0].strip()
            rel = www2fb(items[1].strip())
            label = items[2].strip()
            score = items[3].strip()
            # print("{}   -   {}".format(lineid, rel))
            if lineid in id2rels.keys():
                id2rels[lineid].append( (rel, label, score) )
            else:
                id2rels[lineid] = [(rel, label, score)]
                lineids.append(lineid)
    return lineids, id2rels


# In[11]:

def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)


# In[12]:

def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score =  fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name


# In[13]:

rel_lineids, id2rels = get_relations(train_rel_resultpath)
ent_lineids, id2queries = get_query_texts(train_ent_resultpath)  # ent_lineids may have some examples missing
gold_ent_lineids, id2gold_query = get_query_texts(train_gold_ent_resultpath)  # ent_lineids may have some examples missing


# In[15]:

def get_questions(datapath):
    print("getting questions...")
    id2question = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            sub = items[1].strip()
            pred = items[2].strip()
            obj = items[3].strip()
            question = items[4].strip()
            # print("{}   -   {}".format(lineid, question))
            id2question[lineid] = (sub, pred, question)
    return id2question

datapath = "../data/SimpleQuestions_v2_modified/all.txt"
id2question = get_questions(datapath)


# In[18]:

num_entities_fbsubset = 200000  # 2M - 1959820 , 5M - 1972702
index_ent = get_index(index_entpath)
index_reach = get_index(index_reachpath)
index_names = get_index(index_namespath)


# In[24]:

def calc_tf_idf(question, query, cand_ent_name, cand_ent_count, num_entities, index_ent):
    query_terms = tokenize_text(cand_ent_name)
    doc_tokens = tokenize_text(question)
    common_terms = set(query_terms).intersection(set(doc_tokens))

    # len_intersection = len(common_terms)
    # len_union = len(set(query_terms).union(set(doc_tokens)))
    # tf = len_intersection / len_union
    tf = math.log10(cand_ent_count + 1)
    k1 = 0.5
    k2 = 0.5
    total_idf = 0
    for term in common_terms:
        df = len(index_ent[term])
        idf = math.log10( (num_entities + k1) / (df + k2) )
        total_idf += idf
    return tf * total_idf

def calc_idf(question, cand_ent_name, index_ent):
    query_terms = tokenize_text(cand_ent_name)
    doc_tokens = tokenize_text(question)
    common_terms = set(query_terms).intersection(set(doc_tokens))
    fix_terms = 80000
    total_idf = 0
    for term in common_terms:
        df = len(index_ent[term])
        if df > fix_terms:
            continue # too common term
        idf = math.log10( (fix_terms + 1) / (df + 1) )
        total_idf += idf
    return total_idf

# In[26]:

from collections import defaultdict
data = defaultdict(list)

id2mids = {}
HITS_TOP_ENTITIES = 100
for i, lineid in enumerate(rel_lineids):
    if i % 10000 == 0:
        print("line {}".format(i))

    truth_mid, truth_rel, question = id2question[lineid]
    queries = id2queries[lineid]
    if queries == None or len(queries) == 0:
        queries = [id2question[lineid]]
    C = []
    C_pruned = []
    C_tfidf_pruned = []

    for query in queries:
        query_text = query.lower()  # lowercase the query
        query_tokens = tokenize_text(query_text)
        N = min(len(query_tokens), 3)
        # print("lineid: {}, query_text: {}, relation: {}".format(lineid, query_text, pred_relation))
        # print("query_tokens: {}".format(query_tokens))
        for n in range(N, 0, -1):
            ngrams_set = find_ngrams(query_tokens, n)
            # print("ngrams_set: {}".format(ngrams_set))
            for ngram_tuple in ngrams_set:
                ngram = " ".join(ngram_tuple)
                ngram = strip_accents(ngram)
                # unigram stopwords have too many candidates so just skip over
                if ngram in stopwords:
                    continue
                # print("ngram: {}".format(ngram))
                try:
                    cand_mids = index_ent[ngram]  # search entities
                except:
                    continue
                C.extend(cand_mids)
                # print("C: {}".format(C))
            if (len(C) > 0):
                # print("early termination...")
                break
        # print("C[:5]: {}".format(C[:5]))

        for mid in set(C):
            if mid in index_reach.keys():  # PROBLEM: don't know why this may not exist??
                count_mid = C.count(mid)  # count number of times mid appeared in C
                C_pruned.append((mid, count_mid))

        for mid, count_mid in C_pruned:
            if mid in index_names.keys():
                cand_ent_name = pick_best_name(question, index_names[mid])
                try:
                    truth_name = pick_best_name(question, index_names[truth_mid])
                except:
                    continue
                if cand_ent_name == truth_name:  # if name is correct, we are good
                    data['label'].append(1)
                else:
                    data['label'].append(0)

    #             if mid == truth_mid:
    #                 data['label'].append(1)
    #             else:
    #                 data['label'].append(0)

                data['length_name'].append(len(tokenize_text(cand_ent_name)))
                data['length_question'].append(len(tokenize_text(question)))
                data['length_query'].append(len(query_tokens))
                data['tf'].append(count_mid)
                data['idf'].append(calc_idf(question, cand_ent_name, index_ent))
                data['sques'].append(fuzz.ratio(cand_ent_name, question)/100.0)
                data['squer'].append(fuzz.ratio(cand_ent_name, query_text)/100.0)
                data['pques'].append(fuzz.partial_ratio(cand_ent_name, question)/100.0)
                data['pquer'].append(fuzz.partial_ratio(cand_ent_name, query_text)/100.0)

                C_tfidf_pruned.append((mid, cand_ent_name, data))
        # print("C_tfidf_pruned[:10]: {}".format(C_tfidf_pruned[:10]))

    if len(C_tfidf_pruned) == 0:
        continue

    id2mids[lineid] = C_tfidf_pruned

print("done")



df = pd.DataFrame(data)
print(df['label'].value_counts())

y = df['label']
print(y.head())
X = df.drop('label', axis=1)
print(X.head())


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=0.1)
lr.fit(X, y)

print(lr.score(X, y))
print(lr.coef_)
print(lr.intercept_)

with open('linking_lr.pkl', 'wb') as f:
    pickle.dump(lr, f)



# evaluate on validation set
valid_ent_lineids, valid_id2queries = get_query_texts(valid_ent_resultpath)  # ent_lineids may have some examples missing
valid_gold_ent_lineids, valid_id2gold_query = get_query_texts(valid_gold_ent_resultpath)  # ent_lineids may have some examples missing




