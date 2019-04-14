import pickle
import time

from elasticsearch import Elasticsearch
from googletrans import Translator
from profanityfilter import ProfanityFilter

translator = Translator()
start_time = time.time()

max_e = 50000

"""Builds a backup model (i.e. one that does not keep updating)"""
es = Elasticsearch(['https://9ed43f138f41c3cfbd42ceef576fbeb8.eu-central-1.aws.cloud.es.io:9243'],
                   http_auth=('elastic', '0VkwdsbNK00frpoRFFFlYD97'))

with open("/home/tiina/dev/internet_list.txt", 'r') as f:

    own_list_o=f.readlines()


own_list=[]
for l in own_list_o:
    own_list.append(l.replace('\n', ''))
pf = ProfanityFilter(custom_censor_list=own_list)
print(pf.is_profane('VK'))

how_many = max_e
doc = {
    'size': how_many,

    "query":
        {
            "bool": {
                "must": [{
          "match": {
              "data.origin.country": "JP"
          }
      }, {
          "match": {
             "data.isAd": "false"
          }
      }]
            }
        }
}

res = es.search(
    index='click',
    scroll='1m',
    size=1000,
    body=doc)

sid = res['_scroll_id']
print('TOTAL HITS')
print(res['hits']['total'])
scroll_size = res['hits']['total']
searches = []
counter = 0
q_and_freq=dict()

while (scroll_size > 0) and counter <= how_many:

    print("Scrolling...")
    try:
        res = es.scroll(scroll_id=sid, scroll='1m')
    except:
        break
    # Update the scroll ID
    sid = res['_scroll_id']
    # Get the number of results that we returned in the last scroll
    scroll_size = len(res['hits']['hits'])
    print("scroll size: " + str(scroll_size))
    for re in res['hits']['hits']:


        q=re['_source']['data']['origin']['query']
        print(q)
        try:
            t=translator.translate(q).text
            print(type(t))
            if  pf.is_clean(t) and pf.is_clean(q) :
                with open('di_analysis_clean_all_2xt', 'a') as f:
                    f.write(q+'\n')
                if q not in searches:
                    searches.append(q)
                    counter += 1
                    q_and_freq.update({q:1})
                else:
                    oldfred=q_and_freq[q]
                    q_and_freq.pop(q)
                    q_and_freq.update({q:oldfred+1})
        except:
            print("something went wrong")

    print(counter)

with open('fi_analysiclean.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump(q_and_freq, f, pickle.HIGHEST_PROTOCOL)

with open('fi_analysislean.txt', 'w') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    for s in searches:
        f.write(str(q_and_freq[s]) +' , ' + s + '\n')
