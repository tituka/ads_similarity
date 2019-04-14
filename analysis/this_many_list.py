import SearchModel
from gensim.models import KeyedVectors
word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)

 
backup = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='pulled_pickles/NUpulled0752_010918.pickle')
ll=[]
for uu in backup.item_list:
    ll.append([uu.display_text, uu.this_many])

ss=sorted(ll, key=lambda x:x[1])

with open("word_dist", "a") as f:
    f.write()
