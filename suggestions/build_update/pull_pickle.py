import argparse
import logging
import os
import time
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
import SearchModel
import schedule
import text_tools
from elasticsearch import Elasticsearch
from gensim.models import KeyedVectors

logging.basicConfig(filename="logs/pull_pickle.log", level=logging.ERROR, format="%(asctime)s:%(message)s")
logname='pull'
my_handler = RotatingFileHandler(logname, mode='a', maxBytes=5*1024*1024,
                                 backupCount=2, encoding=None, delay=0)
logger = logging.getLogger(logname)
logger.addHandler(my_handler)

"""Parse command-line arguments"""
ap = argparse.ArgumentParser()
ap.add_argument('-from_days', type=int, default=10)
ap.add_argument('-to_days', type=int, default=0)
ap.add_argument('-min_freq', type=int, default=10)
ap.add_argument('-vectors', type=str, default="vectors/gensim_glove_vectors25.txt")
ap.add_argument('-max_total', type=int, default=50000)
ap.add_argument('-max_unique', type=int, default=10000)
ap.add_argument('-filename', type=str, default='pulled')

args = vars(ap.parse_args())

min_freq = args['min_freq']
max_t = args['max_total']
max_u = args['max_unique']
file_name = args['filename']
word_vectors = KeyedVectors.load_word2vec_format(args['vectors'], binary=False, unicode_errors='ignore')
print("vectors loaded")
time_to=datetime.now() - timedelta(days=args['to_days'])
time_from=datetime.now() - timedelta(days=args['from_days'])

"""Set word vectors todatetime.now() - timedelta(days=100) use, a backup model """
backup = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/gensim_freq.pickle',
    logs=logname)
model=backup
es = Elasticsearch(['https://9ed43f138f41c3cfbd42ceef576fbeb8.eu-central-1.aws.cloud.es.io:9243'],
                   http_auth=('elastic', '0VkwdsbNK00frpoRFFFlYD97'))
many_unique = max_u
many_total=max_t

def remove_oldest_except_last_2(directory):
        """ Removes all files from given list, except the latest two
        """
        path1 = directory
        max_Files = 3

        def sorted_ls(path):
            mtime = lambda f: os.stat(os.path.join(path, f)).st_mtime
            return list(sorted(os.listdir(path), key=mtime))

        del_list = sorted_ls(path1)[0:(len(sorted_ls(path1)) - max_Files)]

        if '.gitignore' in del_list:
            del_list.remove('.gitignore')

        for dfile in del_list:
            os.remove(path1 + dfile)

        logging.debug('Removed '+ ", ".join(del_list))


# noinspection PyBroadException
def update_pickle(search_unique=many_unique, search_total=many_total, t_from=time_from, t_to=time_to, min_f=min_freq):
    doc = {'size': search_total,

        "query": {"bool": {
            "must": [{"match": {"data.isAd": 'true'}}, {"range": {"ts": [{"gte": t_from, "lte": t_to}]}}]}}}

    res = es.search(index='click', scroll='1m', size=1000, body=doc)

    sid = res['_scroll_id']
    logging.debug('TOTAL HITS')
    logging.debug(res['hits']['total'])
    scroll_size = res['hits']['total']
    searches = []
    counter = 0

    while (scroll_size > 0) and counter <= search_total:
        logging.debug("Scrolling...")
        res = es.scroll(scroll_id=sid, scroll='1m')
        # Update the scroll ID
        sid = res['_scroll_id']
        # Get the number of results that we returned in the last scroll
        scroll_size = len(res['hits']['hits'])
        logging.debug("scroll size: " + str(scroll_size))
        for re in res['hits']['hits']:
            counter += 1
            try:
                searches.append(re['_source']['data']['origin']['query'])
            except:
                 logging.debug("No queries found")
        logging.debug(counter)


    print("all")

    print(len(searches))
    print("uniques")
    u_searches = text_tools.remove_dupl(searches)
    print(len(u_searches))
    searches = [x for x in u_searches if searches.count(x) >= min_f]
    print(len(searches))
    print("over 15")

    print(len(searches))
    list_of_files = os.listdir('pulled_pickles/')
    try:
        if not text_tools.has_valid_file(list_of_files, 'pulled_pickles/'):

            logging.debug('No good latest file, must use backup')
            model = SearchModel.SearchModel(raw_texts=searches, w_vectors=word_vectors,
                save_file_name='pulled_pickles/BU' + file_name, max_com=search_unique,

                pickle_2_update='backup_pickles/gensim_freq.pickle', logs=logname, time_to=t_to,
                time_from=t_from)
            logging.debug('successfully updated backup with new model')
        else:
            latest_file = text_tools.latest_good('pulled_pickles/')
            """Case when there is a pulled model to update, try to do so"""
            if latest_file[0]:
                model = SearchModel.SearchModel(raw_texts=searches, w_vectors=word_vectors,
                    save_file_name='pulled_pickles/NU' + file_name, max_com=search_unique, pickle_2_update=latest_file[1],
                    logs=logname, time_to=t_to, time_from=t_from)

                remove_oldest_except_last_2('pulled_pickles/')
                logging.debug('successfully updated newest model')
    except:
        try:
            model = SearchModel.SearchModel(raw_texts=searches, w_vectors=word_vectors,
                save_file_name='pulled_pickles/FN' + file_name, max_com=search_unique, logs=logname, time_to=t_to,
                time_from=t_from)
            remove_oldest_except_last_2('pulled_pickles/')
            logging.debug('loaded fully new model')
        except:
            print("Failed sub first")
            model = backup



"""If model==backup, model hasn't been updated"""
if model == backup:
    logging.debug("Failed")
else:
    logging.debug("Success!")

    """If model==backup, model hasn't been updated"""
    if model == backup:
        logging.debug("Failed")
    else:
        logging.debug("Success!")

    """If model==backup, model hasn't been updated"""
    if model == backup:
        logging.debug("Failed")
    else:
        logging.debug("Success!")


"""Build first model when program is started"""
update_pickle()
"""Set here time interval for updating file"""
schedule.every(300).minutes.do(update_pickle)

while 1:
    schedule.run_pending()
    time.sleep(5)

