import Company_system
from gensim.models import KeyedVectors

word_vectors= KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
system=Company_system.Company_system(w_vectors=word_vectors, pickle_file='backup_pickles/backup.pickle')

from newsapi import NewsApiClient


newsapi = NewsApiClient(api_key='172b8e9311e44e39abf9c18ba9c5f7c2')


news=newsapi.get_top_headlines(language='en', country='us')
print(news.keys())

tandd=[]
for n in news['articles'][:10]:
    print(n['title'])
    print(n['description'])
    system.similar_by_text(max([n['description'], n['title']], key=len), cosine=True)
