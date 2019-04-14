import glob
import math
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wikipedia
from nltk import word_tokenize, ngrams
from nltk.corpus import reuters
from numpy import argsort, sqrt
from profanityfilter import ProfanityFilter
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

"""List of prohibited words"""
print( os.getcwd())

with open("misc_files/internet_list.txt", 'r') as f:
    own_list_o=f.readlines()


with open("misc_files/all_dict.pickle", "rb") as f:
	avg_dict=pickle.load(f)

with open("misc_files/prepositions.txt", "r") as f:
    prep=f.readlines()
    prepositions=[]
    for p in prep:
        prepositions.append(p.rstrip('\r\n'))

average_freq=np.mean(list(avg_dict.values()))

own_list=[]
for l in own_list_o:
    own_list.append(l.replace('\n', ''))
pf = ProfanityFilter(custom_censor_list=own_list)

def is_allowed(sstring):
    """returns True if sstring does not run fowl of profanity filter based on above list"""
    return (pf.is_clean(sstring))

def tf(word, token_list):
    """"Returns term frequency"""
    return token_list.count(word) / len(token_list)

def n_containing(word, token_list_list):
    """Returns how often a word appears in a list of list of tokens"""
    return sum(1 for token_list in token_list_list if word in token_list)

def idf(word, token_list_list):
    """Returns the inverse document frequecy of word in token_list_list"""
    return math.log(len(token_list_list) / (1 + n_containing(word, token_list_list)))

def tfidf(word, token_list, token_list_list):
    """Returns term frwquency times inverse document frequency"""
    return tf(word, token_list) * idf(word, token_list_list)

def knn_search(search_item, D, cosine=False, furthest=False, K=5):
    """ gives the indices of the K nearest vectors to the sum vector of data among D """
    # euclidean distances from the other points
    if cosine:
        sqd = flat_list(flat_list(
            [cosine_similarity(np.asmatrix(search_item.sum_vec), np.asmatrix(y)) for y in [z.sum_vec for z in D]]))
        idx = argsort(sqd)
    else:
        sqd = [sqrt(((y - (search_item.sum_vec)) ** 2).sum(axis=0)) for y in [z.sum_vec for z in D]]
        idx = argsort(sqd)[::-1]  # sorting
        # return the indexes of K nearest neighbours
    if furthest:
        idx = idx[::-1]
    return_list = []
    for idn in idx:
        return_list.append(D[idn])
    if search_item in return_list:
        return_list.remove(search_item)
    return return_list[::-1][:K]


def knn_search_vector(vector, D, cosine=False, furthest=False, K=5):
    """ find K nearest neighbours of data among D """
    print("K imputted:" + str(K))
    # euclidean distances from the other points
    if cosine:
        sqd = flat_list(
            flat_list([cosine_similarity(np.asmatrix(vector), np.asmatrix(y)) for y in [z.sum_vec for z in D]]))
        idx = argsort(sqd)
    else:
        sqd = [sqrt(((y - vector) ** 2).sum(axis=0)) for y in [z.sum_vec for z in D]]
        idx = argsort(sqd)[::-1]  # sorting
    # return the indexes of K nearest neighbours
    # if furthest:
    #    idx=idx[::-1]
    return_list = []
    for idn in idx:
        return_list.append(D[idn])
    return return_list[::-1][:K]


def knn_s_all_vector(vector, D, cosine=False, furthest=False, K=5):
    """ find K nearest neighbours of data among D """
    # euclidean distances from the other points
    print("VEC K imputted:" + str(K))
    if cosine:
        sqd = flat_list(flat_list([cosine_similarity(np.asmatrix(vector), np.asmatrix(y)) for y in [z for z in D]]))
        idx = argsort(sqd)[::-1]
    else:
        sqd = [sqrt(((y - vector) ** 2).sum(axis=0)) for y in [z for z in D]]
        idx = argsort(sqd)  # sorting
    # return the indexes of K nearest neighbours
    if furthest:
        idx = idx[::-1]
    return idx[:K]


def flat_list(ll):
    return [item for sublist in ll for item in sublist]


def tsne_plot(searches_to_plot, model):
    "Creates and TSNE model and plots it"
    labels = [search_item.title() for search_item in searches_to_plot]
    w_vectors = [company.sum_vector() for company in searches_to_plot]

    tsne_model = TSNE(perplexity=10, n_components=2, init='pca', n_iter=5000, random_state=23)
    new_values = tsne_model.fit_transform(w_vectors)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()


def similarity(comp_1, comp_2, cosine=True):
    if cosine:
        return ((cosine_similarity(np.asmatrix(comp_1.sum_vec), np.asmatrix(comp_2.sum_vec)))[0][0])
    else:
        return (np.linalg.norm(comp_1.sum_vec - comp_2.sum_vec))


def similarity_vec(vec1, vec2, cosine=True):
    if cosine:
        return ((cosine_similarity(np.asmatrix(vec2), np.asmatrix(vec1)))[0][0])
    else:
        return (sqrt((vec1 - vec2) ** 2).sum(axis=0))


def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dotproduct(v, v))


def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))


def project_vector(x, y):
    print(np.linalg.norm(x) * math.cos(angle(x, y)))
    return np.linalg.norm(x) * math.cos(angle(x, y))


def project(compx, compy):
    x = compx.sum_vec
    y = compy.sum_vec
    return np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2 - 2 * x * y * math.cos(x, y)


def print_comparison(self, search_item, compare_items, cosine=False, top_n=5):
    print("print top imputted:" + str(top_n))
    if cosine:
        abbr = " S:"
    else:
        abbr = ' D:'
    print("Given text \'" + search_item.display_text + '\'' + " is close to:")
    for item in compare_items:
        print('\n' + str(compare_items.index(item) + 1) + '. ' + abbr + str(
            format(similarity(search_item, item, cosine), '3.2f')) + item.display_text)


def print_comparison_vec(self, search_vec, compare_items, cosine=False, top_n=5):
    print("VEC print top imputted:" + str(top_n))
    if cosine:
        abbr = " S:"
    else:
        abbr = ' D:'
    print("Given vector is close to:")
    for item in compare_items:
        print('\n' + str(compare_items.index(item) + 1) + '. ' + item.display_text.upper() + '[' + str(
            self.item_list.index(item)) + ']' + abbr + str(
            format(similarity_vec(search_vec, item.sum_vec, cosine), '3.2f')) + '\n' + item.display_text)

def tokenize_lower(string_to_process):
    """Returs given string tokenized and in lowercase"""

    return [x.lower() for x in word_tokenize(string_to_process)]

def good_pickle(model_pickle):
    """returns True if model is able to load and contains at least one entry"""
    try:
        with open(model_pickle, 'rb') as f:
            loadedpickle = pickle.load(f)
            try_item_list =loadedpickle

    except:
        print('Cannot load ' + model_pickle)
        return False
    if len(try_item_list)==0:
        print(model_pickle + 'is an empty list')
        return False
    else:
        return True


def latest_good(directory:list):
    """Returns True if directory has a file which good_pickle returns as True, as well as that file. If no file is a
    good_pickle, returns false, as well as the backup pickle file"""
    list_of_files = glob.glob(directory+'*')  # * means all if need specific format then *.csv
    while (not list_of_files==None) and (not list_of_files==[]) and (not list_of_files==[directory+'.gitignore']):
        latest_file = max(list_of_files, key=os.path.getctime)
        print( latest_file)
        if good_pickle(latest_file):
            return True, latest_file
        else:
            list_of_files.remove(latest_file )
    return([False, 'backup_pickles/gensim_freq.pickle'])


def has_valid_file(should_be_files:list, directory:str):
    """Returns True is files in list should_be_file in given directory is a good pickle. If none such files are
    found, returns False"""
    if should_be_files==None or should_be_files==[] or should_be_files==['.gitignore']:
        return False
    else:
        for f in should_be_files:
            if good_pickle(directory+f):
                return True
    return False

def find_compound(token:str, ind:int,  token_list:list, word_vectors):
    """Returns a compound token made up of token and the next after that, when one is found in the word vector model"""
    longer=False
    newt = token
    n=1
    all_longs=[x for x in all_concat(token_list[ind:]) if x[0] in word_vectors]
    if not all_longs==[]:
        best_all=max(all_longs, key=lambda x: x[1])
        best_token=best_all[0]
        best_len=best_all[1]
    else:
        best_token = token
        best_len = 1
    return best_token, best_len

def replace_with_multi(token_list, word_vectors):
    """Replaces single_word tokens with compound ones when they are found in the model. Addtionally, removes tokens
    which are not in given set of word_vectors"""

    new_list=[]
    workn=0
    n=1
    nextgood=0
    for count, item in enumerate(token_list):
        if count<nextgood:
            pass
        else:
            newc=find_compound(item, count, token_list, word_vectors)
            new_list.append(newc[0])
            nextgood=count+newc[1] +1
    new_list=[x for x in new_list if x in word_vectors]
    return new_list

def all_concat(token_list):
    all_list=[]
    for i in range(len(token_list)):
        all_list.append([('_').join(token_list[:i+1]), i])
    return(all_list)

def process_multi(tokens_in_in_string, word_vectors):
    tokenized=[w.lower() for w in word_tokenize(str(tokens_in_in_string))]
    return(replace_with_multi(tokenized, word_vectors))

def are_repetitions(disp_string):
    tokens=[x.lower() for x in word_tokenize(disp_string) if x.isalpha()]
    for count, item in enumerate(tokens):
        if count<len(tokens)-1:
            if tokens[count]==tokens[count+1]:
                return True
    return False


def missing_in_model(pötkö, n, word_vectors):
    try:
        final_q=wikipedia.page(pötkö)
        ll=[x for x in final_q.categories if not (x.startswith("Articles") or "articles" in x)]
        if pötkö.lower() not in word_vectors:
            return True, ll
    except:
        grams = []
        for m in range(2,n+1):
            grams+=ngrams(pötkö.split(), m)
        #print((grams))
        wp_list=[]
        wiki_pages=[]
        has_wiki=[]
        for gram in grams:
            #print(gram)
            try:
                wp_list.append(wikipedia.page(gram))
                wiki_pages.append(True)
                has_wiki.append(gram)
            except:
                print("whatever")
        lonkerot=[]
        for count, item in enumerate(wiki_pages):
            if item:
                print("_".join(has_wiki[count]) )
                if "_".join(has_wiki[count]) in word_vectors:
                        print("found")
                        return False, []
                else:
                    print("not found")
        for w in wp_list:
            if len(has_wiki[count]) == max([len(x) for x in has_wiki]):
                lonkerot.append(wp_list[count])
        lonkerot=[lon.categories for lon in lonkerot ]
        lyhyt_lonkerot=[]
        for lon in lonkerot:
            lyhyt_lonkerot.append([x for x in lon if not (x.startswith("Articles") or "articles" in x)])

        if lyhyt_lonkerot==[]:
            return False, []
        return True, lyhyt_lonkerot[0]


def get_wiki_categories(hölympöly, word_vectors):
    if len([x for x in word_tokenize(hölympöly) if x.isalpha()]) < 2:
        return hölympöly
    check_wiki = missing_in_model(hölympöly, 3, word_vectors)
    if check_wiki[0]:
        cats=" ".join(check_wiki[1])
        with open("logs/wiki_cat_texts.txt", "a") as f:
            f.write(hölympöly +" "+cats + "\n")
        return cats
    else:
        with open("logs/no_wiki.txt", "a") as f:
            f.write(hölympöly + "\n")
        return hölympöly

def reuters_t_l_l():
    token_list_list=[]
    for file_id in reuters.fileids():
        token_list_list.append(reuters.words(file_id))
    return token_list_list

def t_l_l_to_dict_and_total(t_l_l: list):
    small_tll=[]
    for t_l in t_l_l:
        small_tll.append([x.lower() for x in t_l])
    small_flat = flat_list(small_tll)
    freq_dict=dict()
    counter=0
    for word in list(set( small_flat)):
        total_freq= small_flat.count(word)
        in_doc=0
        for t_l in small_tll:
            if word in t_l:
                in_doc+=1
        freq_dict.update({word:(total_freq, in_doc)})
        print(counter)
        counter+=1
    return(freq_dict, len(t_l_l))

def other_tll(word, t_l, tfidf_dict, tfidf_total):
    if word not in tfidf_dict:
        return ((t_l.count(word)/len(t_l)))*(tfidf_total/average_freq+2)
    return ((t_l.count(word)/len(t_l)))*(tfidf_total/tfidf_dict[word])

def reuters_tfidf(word, t_l):
    my_file = Path("misc_files/words_freqs.pickle")

    with open("misc_files/words_freqs.pickle", "rb") as pickle_file:
        aa = pickle.load(pickle_file)
        r_dic, r_len = t_l_l_to_dict_and_total(aa)
    return other_tll(word, t_l, r_dic, r_len)

def remove_dupl(raw_strings):
    seen=[]
    for raw_string in raw_strings:
        if raw_string not in seen:
            seen.append(raw_string)
    return seen

def reuters_tfidf(word, t_l):
    my_file = Path("misc_files/words_freqs.pickle")

    with open("misc_files/words_freqs.pickle", "rb") as pickle_file:
        aa = pickle.load(pickle_file)
        r_dic, r_len = t_l_l_to_dict_and_total(aa)
    return other_tll(word, t_l, r_dic, r_len)

def nb_r_tfidf(word, t_l):
    """Returns the tf-idf score using frequencies from the average of some wikipedia categories and reuters corpora
    works ideally with numberbatch vectors"""
    return other_tll(word, t_l, avg_dict, 10788)

def process_for_display(tokens):
    final_text=""
    for token in tokens:
        if token not in prepositions:
            final_text+= " " + token.capitalize()
        else:
            final_text+=" " + token
    return final_text[1:]