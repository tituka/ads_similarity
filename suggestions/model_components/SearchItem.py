# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 19:35:13 2018

@author: tkalliom
"""
from collections import Counter
from datetime import datetime
import os
import SearchModel
import numpy as np
import sys
import sys
sys.path.insert(0, "tools")
import text_tools
from elasticsearch import Elasticsearch
from nltk.tokenize import word_tokenize


class SearchItem:
    # list of list of tokens in company description:
    token_list_list = []
    vocabulary = []
    es = Elasticsearch(['https://9ed43f138f41c3cfbd42ceef576fbeb8.eu-central-1.aws.cloud.es.io:9243'],
        http_auth=('elastic', '0VkwdsbNK00frpoRFFFlYD97'))


    def __init__(self, model:SearchModel, raw_text:str, display_text=None, used_wiki=False):
        self.used_wiki=used_wiki
        if display_text==None:
            self.display_text = raw_text.rstrip('\r\n')
        else:
            self.display_text=display_text.rstrip('\r\n')

        self.model=model
        SearchItem.token_list_list = self.model.t_l_l
        self.word_vectors = self.model.word_vectors
        self.time_from = self.model.time_from
        self.time_to = self.model.time_to
        self.raw_text = raw_text
        self.count_in_time_frame()
        self.lower_tokens = []
        self.used_tokens=text_tools.process_multi(raw_text, model.word_vectors)

        self.token_vectors, self.used_tokens = self.get_vectors()
        self.weighed_vectors, self.weights_sum, self.token_weights = self.get_weighed_vectors()
        self.sum_vec = self.get_sum_vec()
        for token in word_tokenize(str(self.display_text)):
            self.lower_tokens.append(token.lower())
        self.modified = False
        #  if (self.company_index%50==0):

    def count_in_time_frame(self):
        doc = {'size': 1000, "query": {"bool": {"must": [{"match": {"data.origin.query.keyword": self.display_text}},
            {"match": {"data.isAd": 'true'}}]}
        }}
        res = SearchItem.es.search(index='click', scroll='1m', size=1000, body=doc)
        self.this_many = res['hits']['total']

    def get_vectors(self):
        """gets word vectors for each token found in vectors from self.raw_desc"""
        t_used = []
        token_vectors = []
        #print(list(set(self.lower_tokens)))
        list_set = text_tools.process_multi(self.raw_text, self.word_vectors)
        #print(list_set)
        for count, item in enumerate(list_set):
            if item in self.model.word_vectors and any(c.isalpha() for c in item):
                #print(item)
                token_vectors.append(self.word_vectors[item])
                t_used.append(item)
        return token_vectors, t_used

    def get_weighed_vectors(self):
        """Returns vectors multiplied by appropriate sum according to tf-idf, as well as the sum of
        these vectors, and the indiviudual tf-idf weights for each vector"""
        w_vectors = []
        w_sum = 0
        token_weight = dict()

        for count, item in enumerate(self.token_vectors):
            tdi = text_tools.nb_r_tfidf(self.used_tokens[count], self.used_tokens)
            w_vectors.append(tdi * item)
            w_sum += tdi
            token_weight.update({self.used_tokens[count]: tdi})
        return w_vectors, w_sum, token_weight

    def get_sum_vec(self):
        if self.weights_sum == 0:
            return np.zeros(self.model.word_vectors.vector_size)
        else:
            return (1 / self.weights_sum) * sum(self.weighed_vectors)

    def find_longest(self, token, token_list, place_in_list):
        """"Returns longest compound vector that can be found in given token list, the fist word in compound at
        index place_in_list. Also returns number of tokens used in compound"""
        n = place_in_list
        newt = token
        while n < len(token_list) - 1:
            newt += '_' + token_list[n + 1]
            if newt in self.model.word_vectors:
                n += 1
            else:
                n = len(token_list)
        return newt, n

    def reset(self):
        """"Resets value of modified (changed in)"""
        self.sum_vec= self.get_sum_vec()
        self.modified=False

    def change_time_frame(self, time_from:datetime, time_to:datetime):
        """Changes the timeframe of object to time_from, time_to"""
        self.time_from = time_from
        self.time_to = time_to
        self.count_in_time_frame()
        return self

