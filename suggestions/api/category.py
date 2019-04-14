import os
print( os.getcwd())
import wikipedia
import sys
sys.path.insert(0, "model_components")
import SearchModel
from gensim.models import KeyedVectors
import operator
sys.path.insert(0, "tools")

import text_tools

word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
model = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/backup.pickle')
"""
with open("small_list.txt", "r") as f:
    raw=f.read()

list=raw.split("\n")
s_items=[]
for item in list:
    s_items.append(model.try_search(item, use_wiki=True))


"""
sys.path.insert(0, "analysis")



sports=model.try_search("sports, atheletics, drafting, athlete, league", display="sports", use_wiki=False)
beauty = model.try_search("beauty, fashion, cosmetics, clothing, design", display="beauty", use_wiki=False)
film = model.try_search("film, movies, showtimes, sequal, trilogy, cinema, actor, actress, director ",
    display="film", use_wiki=False)
tech = model.try_search("technology, science, innovation, breakthrough, research ", display="tech", use_wiki=False)
internet_people = model.try_search("youtuber, blogger, internet_personality, influencer, instagram",
    display="internet people", use_wiki=False)
news = model.try_search("news, world_events, politics, diplomacy, disaster, weather", display="news", use_wiki=False)
business = model.try_search("business, finance, stock_exchange, investing", display="business", use_wiki=False)
gaming= model.try_search("e-sports, gaming, gamer, playstation, xbox", display="gaming",use_wiki=False)
music= model.try_search("music, musician, album, singer, producer, dj, concert, rapper, mixtape, hip hop",
     display="music", use_wiki=False)

categories=(sports, beauty, film, tech, internet_people, news  , business, gaming, music)
"""
full=dict()
for s in s_items:
    comp_dict = dict()
    for cat in categories:
        comp_dict.update({cat: text_tools.similarity(cat, s, cosine=True)})
    full.update({text_tools.process_for_display(s.lower_tokens) :max(comp_dict.items(), key=operator.itemgetter(1))[0]})
   # print(text_tools.process_for_display(s.lower_tokens) + ": " + max(comp_dict.items(), key=operator.itemgetter(
# 1))[0].display_text + "\n")
print('----------------------------------------------------------')
print(len(full.keys()))



with open("file", "w") as f:
    for x in full.keys():
        print(x + ": " + full[x].display_text + "\n")
        f.write(x +": " + full[x].display_text +"\n")
"""

def category_and_score(s_string:str):
    s=model.try_search(s_string, use_wiki=True)
    comp_dict = dict()
    for cat in categories:
        comp_dict.update({cat: text_tools.similarity(cat, s, cosine=True)})
    return  max(comp_dict.items(), key=operator.itemgetter(1))[0].lower_tokens, comp_dict[max(comp_dict.items(),
        key=operator.itemgetter(1))[0]]

def category_from_summary(s_string:str):
    try:
        page=wikipedia.page(s_string)
        summary=page.summary
    except:
        summary=s_string
    comp_dict = dict()
    s=model.try_search(summary, display=summary, use_wiki=False)
    for cat in categories:
        comp_dict.update({cat: text_tools.similarity(cat, s, cosine=True)})
    return  max(comp_dict.items(), key=operator.itemgetter(1))[0].lower_tokens, comp_dict[max(comp_dict.items(),
        key=operator.itemgetter(1))[0]]
