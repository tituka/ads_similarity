# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:41:35 2018

@author: tkalliom
"""

import argparse
import os
import string
import sys
from contextlib import contextmanager
sys.path.insert(0, "model_components")
import SearchModel
sys.path.insert(0, "tools")
import text_tools
from gensim.models import KeyedVectors

# ap = argparse.ArgumentParser()
# ap.add_argument('-vectors', type=str, default="vectors/gensim_glove_vectors25.txt")
# args = vars(ap.parse_args())



file_dir = os.path.dirname('/home/tiikal/ml')
sys.path.append(file_dir)
word_vectors = KeyedVectors.load_word2vec_format('vectors/gensim_glove_vectors25.txt', binary=False,
unicode_errors='ignore')
#word_vectors = KeyedVectors.load_word2vec_format("vectors/numberbatch-en-17.06.txt", binary=False,
    # unicode_errors='ignore')


#word_vectors = KeyedVectors.load_word2vec_format(args['vectors'], binary=False, unicode_errors='ignore')


model = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/gensim_freq.pickle')

def clean_string(input_str):
    return (''.join(filter(lambda x: x in string.printable, input_str)))


@contextmanager
def update_after():
    yield
    if model.should_update(2):
        if model.is_newest_updateable():
            model.change_pickle(model.newest_model())
            print ('CHANGED MODEL TO ' + model.current_file)
        else:
            print('model stayed the same:' + model.current_file)
    else:
        print('model up to date')

def similar(sisaan, maxl, countc):
    with update_after():
        similar_output = model.similar_by_text(sisaan, cosine=True, max_l=maxl, count_c=countc)
        output_list = []
        for count, item in enumerate(similar_output):
            keys = ['frequency', 'rank', 'similarity', 'text']

            values = [item.this_many, str(count + 1),
                str(text_tools.similarity(model.try_search(sisaan, use_wiki=True), item)),  text_tools.process_for_display(item.lower_tokens)]

            output_sub_dict = dict(zip(keys, values))
            output_list.append(output_sub_dict)
            print('company enumerated!')
            print(item.raw_text)
            print(item.display_text)
        print(output_list)

        return output_list

    similar_output = model.similar_by_text(sisaan, cosine=True, top_n=3)
    print(sisaan)
    output_list = []
    for count, item in enumerate(similar_output):
        keys = ['index', 'rank', 'distance', 'text', 'frequency']
        similarity_score = text_tools.similarity(model.try_search(sisaan), item)
        if similarity_score > 0:
            values = [str(model.get_index(item)), str(count + 1),
                str(similarity_score), text_tools.process_for_display(item.lower_tokens), item.this_many]
        else:
            values = [str(model.get_index(item)), str(count + 1), str(similarity_score),'',
                '']
        output_sub_dict = dict(zip(keys, values))
        output_list.append(output_sub_dict)
        print('company enumerated!')
    print(output_list)

    return output_list


def compare_two(sisaan1, sisaan2, cosine=True):
    print(model.similar_two_text(sisaan1, sisaan2, cosine))
    return (model.similar_two_text(sisaan1, sisaan2, cosine))