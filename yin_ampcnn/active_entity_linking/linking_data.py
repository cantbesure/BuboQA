#!/usr/bin/python

import os
import sys
import argparse
import pickle
import math
import pandas as pd

from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from collections import defaultdict

"""
Example command to run program:
python linking.py -d ../data/SimpleQuestions_v2_augmented/all.txt --index_ent ../indexes/entity_2M.pkl \
--index_reach ../indexes/reachability_2M.pkl --index_names ../indexes/names_2M.pkl \
--ent_result query-text/ -n 20 --output linking-results
"""
stopwords = set(stopwords.words('english'))

def www2fb(in_str):
    if in_str.startswith("www.freebase.com"):
        return 'fb:%s' % (in_str.split('www.freebase.com/')[-1].replace('/', '.'))
    return in_str

def get_index(index_path):
    print("loading index from: {}".format(index_path))
    with open(index_path, 'rb') as f:
        index = pickle.load(f)
    return index


def get_query_texts(ent_resultpath):
    print("getting query texts...")
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


def find_ngrams(input_list, n):
    ngrams = zip(*[input_list[i:] for i in range(n)])
    return set(ngrams)

def get_questions(datapath):
    print("getting questions...")
    id2question = {}
    with open(datapath, 'r') as f:
        for line in f:
            items = line.strip().split("\t")
            lineid = items[0].strip()
            ent = items[1].strip()
            ent_name = items[2].strip()
            rel = items[3].strip()
            obj = items[4].strip()
            question = items[5].strip()
            # print("{}   -   {}".format(lineid, question))
            id2question[lineid] = (ent, ent_name, rel, question)
    return id2question


def pick_best_name(question, names_list):
    best_score = None
    best_name = None
    for name in names_list:
        score = fuzz.ratio(name, question)
        if best_score == None or score > best_score:
            best_score = score
            best_name = name

    return best_name

def calc_tf_idf(question, query, cand_ent_name, cand_ent_count, num_entities, index_ent):
    query_terms = cand_ent_name.split()
    doc_tokens = question.split()
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
    query_terms = cand_ent_name.split()
    doc_tokens = question.split()
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


def linking_data_one_file(id2question, index_ent, index_reach, index_names, ent_resultpath):
    ent_lineids, id2queries = get_query_texts(ent_resultpath)  # ent_lineids may have some examples missing
    id2mids = {}
    data = defaultdict(list)

    for i, lineid in enumerate(ent_lineids):
        if not lineid in id2question.keys():
            continue

        if i % 1000 == 0:
            print("line {}".format(i))

        truth_mid, truth_mid_name, truth_rel, question = id2question[lineid]
        queries = id2queries[lineid]
        C = []  # candidate entities
        C_counts = []
        C_scored = []

        for query_text in queries:
            query_tokens = query_text.split()
            N = min(len(query_tokens), 3)

            for n in range(N, 0, -1):
                ngrams_set = find_ngrams(query_tokens, n)
                # print("ngrams_set: {}".format(ngrams_set))
                for ngram_tuple in ngrams_set:
                    ngram = " ".join(ngram_tuple)
                    # unigram stopwords have too many candidates so just skip over
                    if ngram in stopwords:
                        continue
                    # print("ngram: {}".format(ngram))
                    try:
                        cand_mids = index_ent[ngram]  # search entities
                    except:
                        continue
                    C.extend(cand_mids)
                if (len(C) > 0):
                    break #early termination

            for mid in set(C):
                count_mid = C.count(mid)  # count number of times mid appeared in C
                C_counts.append((mid, count_mid))

            for mid, count_mid in C_counts:
                if mid in index_names.keys():
                    cand_ent_name = pick_best_name(question, index_names[mid])
                    try:
                        truth_name = pick_best_name(question, index_names[truth_mid])
                    except:
                        continue
                    if cand_ent_name == truth_name:  # if name is correct, we are good
                        data['exact_name_match'].append(1)
                    else:
                        data['exact_name_match'].append(0)
                        if fuzz.ratio(cand_ent_name, truth_name) >= 60:
                            data['partial_name_match'].append(1)
                        else:
                            data['partial_name_match'].append(0)

                    if mid == truth_mid:
                        data['true_label'].append(1)
                    else:
                        data['true_label'].append(0)
                    data['lineid'] = lineid
                    data['query'] = query_text
                    data['length_name'].append(len(cand_ent_name.split()))
                    data['length_question'].append(len(question.split()))
                    data['length_query'].append(len(query_tokens))
                    data['tf'].append(count_mid)
                    data['idf'].append(calc_idf(question, cand_ent_name, index_ent))
                    data['sques'].append(fuzz.ratio(cand_ent_name, question) / 100.0)
                    data['squer'].append(fuzz.ratio(cand_ent_name, query_text) / 100.0)
                    data['pques'].append(fuzz.partial_ratio(cand_ent_name, question) / 100.0)
                    data['pquer'].append(fuzz.partial_ratio(cand_ent_name, query_text) / 100.0)

    df = pd.DataFrame(data)
    return df

def active_entity_linking(data_path, index_entpath, index_reachpath, index_namespath, ent_resultdir, outpath):
    id2question = get_questions(data_path)
    index_ent = get_index(index_entpath)
    index_reach = get_index(index_reachpath)
    index_names = get_index(index_namespath)

    fnames = ["train", "valid", "test"]
    for fname in fnames:
        inpath = os.path.join(outpath, fname + ".txt")
        ent_resultpath = os.path.join(ent_resultdir, fname + ".txt")
        outfname = os.path.join(outpath, "{}-h{}-data.pkl".format(fname))
        df = linking_data_one_file(id2question, index_ent, index_reach, index_names, ent_resultpath)
        df.to_pickle(outfname)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Do entity linking')
    parser.add_argument('-d', '--dataset', dest='dataset', action='store', required=True,
                        help='path to the AUGMENTED dataset all.txt file')
    parser.add_argument('--index_ent', dest='index_ent', action='store', required=True,
                        help='path to the pickle for the inverted entity index')
    parser.add_argument('--index_reach', dest='index_reach', action='store', required=True,
                        help='path to the pickle for the graph reachability index')
    parser.add_argument('--index_names', dest='index_names', action='store', required=True,
                        help='path to the pickle for the names index')
    parser.add_argument('--ent_result', dest='ent_result', action='store', required=True,
                        help='path to the entity detection directory that contains results with the query texts')
    parser.add_argument('--output', dest='output', action='store', required=True,
                        help='directory path to the data of entity linking')

    args = parser.parse_args()
    print("Dataset: {}".format(args.dataset))
    print("Index - Entity: {}".format(args.index_ent))
    print("Index - Reachability: {}".format(args.index_reach))
    print("Index - Names: {}".format(args.index_names))
    print("Entity Detection Results: {}".format(args.ent_result))
    print("Output: {}".format(args.output))
    print("-" * 80)

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    active_entity_linking(args.dataset, args.index_ent, args.index_reach, args.index_names, args.ent_result, args.output)

    print("Active Entity Linking done.")
