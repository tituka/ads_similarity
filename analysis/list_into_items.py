import argparse
from gensim.models import KeyedVectors
import os
import sys
sys.path.insert(0, "model_components")
import SearchModel
sys.path.insert(0, "tools")
import text_tools
model = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/backup.pickle')


"""Parse command-line arguments"""
ap = argparse.ArgumentParser()
ap.add_argument('-vectors', type=str, default='vectors/vectors/numberbatch-en.txt.txt')
ap.add_argument('-filename', type=str, default='misc_files/sports.txt')

args = vars(ap.parse_args())

file_name = args['filename']
word_vectors = KeyedVectors.load_word2vec_format(args['vectors'], binary=False, unicode_errors='ignore')

with open(file_name, "r") as f:
    raw_lines=f.readlines()

processed_lines=[]
for line in raw_lines:
    processed_lines.append(line.replace(" \n", "").replace("\n"))

line_items=[]
for p_line in processed_lines:
    line_items.append(line_items.append(model.try_search(p_line, use_wiki=True)))

with open(file_name +".pickle", 'wb') as f:
    pickle.dump(line_items.item_list, f, pickle.HIGHEST_PROTOCOL)