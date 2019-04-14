import argparse
import logging
import time
from elasticsearch import Elasticsearch
from gensim.models import KeyedVectors
from datetime import datetime, timedelta
import os
import sys
sys.path.insert(0, "model_components")
import SearchModel
sys.path.insert(0, "tools")
import text_tools

logging.basicConfig(filename="logs/build_backup.log", level=logging.ERROR, format="%(asctime)s:%(message)s")
logname='backup'
logger = logging.getLogger(logname)
start_time = time.time()


time_to=datetime.now()
time_from=datetime.now() - timedelta(days=700)

def count_in_time_frame(es, sear, time_from, time_to):
    doc = {'size': 1000, "query": {"bool": {
        "must": [{"match": {"data.origin.query.keyword": sear}}, {"match": {"data.isAd": 'true'}}, {"range": {"ts": [{"gte": time_from, "lte": time_to}]}}]}}}
    res = es.search(index='click', scroll='1m', size=1000, body=doc)
    print(res['hits']['total'])
    return res['hits']['total']

def get_searches(how_many, time_from, time_to):
    print("getting searches")

    if how_many < 1000:
        resnumner = how_many
    else:
        resnumner = 1000

    es = Elasticsearch(['https://9ed43f138f41c3cfbd42ceef576fbeb8.eu-central-1.aws.cloud.es.io:9243'],
                       http_auth=('elastic', '0VkwdsbNK00frpoRFFFlYD97'))

    doc = {
        'size': how_many,
    
        "query": {"bool": {
            "must": [{"match": {"data.isAd": 'true'}}]}}}


    res = es.search(
        index='click',
        scroll='1m',
        size=resnumner,
        body=doc)
    
    sid = res['_scroll_id']
    logging.debug('TOTAL HITS')
    logging.debug(res['hits']['total'])
    scroll_size = res['hits']['total']
    searches = []
    counter = 0
    
    while (scroll_size > 0) and counter < how_many:
        print(str(counter) +"---------------------------------")

        logging.debug("Scrolling...")
        res = es.scroll(scroll_id=sid, scroll='1m')
        # Update the scroll ID
        sid = res['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(res['hits']['hits'])
        logging.debug("scroll size: " + str(scroll_size))
        for re in res['hits']['hits']:

            try:

                searches.append(re['_source']['data']['origin']['query'])
                logging.debug(re['_source']['data']['origin']['query'])
                counter += 1
            except:
                logging.debug("No queries found")
        logging.debug(counter)
        print("got searches")
    print("all")

    print(len(searches))
    print("uniques")
    u_searches = text_tools.remove_dupl(searches)
    print(len(u_searches))
    searches = [x for x in u_searches if searches.count(x) >= 7]
    print(len(searches))
    print("over 10")

    print(len(searches))

    return searches
    
def main(args):

    """Parse command-line arguments"""
    ap = argparse.ArgumentParser()
    ap.add_argument('-vectors', type=str, default='vectors/gensim_glove_vectors25.txt')
    ap.add_argument('-max_total', type=int, default=10000)
    ap.add_argument('-max_unique', type=int, default=100)
    ap.add_argument('-filename', type=str, default='backup')

    args = vars(ap.parse_args())

    max_t = args['max_total']
    max_u = args['max_unique']
    file_name = args['filename']
    word_vectors = KeyedVectors.load_word2vec_format(args['vectors'], binary=False, unicode_errors='ignore')
    print("vectors loaded")
    searches=get_searches(max_t, time_from, time_to)
    print("building it")
    model = SearchModel.SearchModel(raw_texts=searches,w_vectors=word_vectors,
                                            save_file_name='backup_pickles/' + file_name, max_com=max_t,
                                            date_stamp=False, )




if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))