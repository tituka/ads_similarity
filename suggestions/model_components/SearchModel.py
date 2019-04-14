import glob
import logging
import operator
import os
import pickle
import time
from datetime import timedelta, datetime
import SearchItem
import sys
cwd = os.getcwd()
print(cwd)
sys.path.insert(0, "tools")
import numpy as np
cwd = os.getcwd()
print(cwd)
import text_tools
from nltk import ngrams as ng
from nltk.tokenize import word_tokenize
from numpy import sqrt, argsort

""""SearchModel is an updateable model made up of search entries and their vectors as objects of type SearchItem"""


class SearchModel:

    def __init__(self, raw_texts: list = None, w_vectors=None, cosine: bool = False, pickle_file=None,
            save_file_name: str = None, max_com: int = 5000, pickle_2_update: str = None, date_stamp=True, logs=None,
            time_from: int = 14, time_to: datetime = datetime.now()):
        """ Initialized a company system. If pickle_file==None, creates a new
        system based on given list of company titles and descriptions, using give word vectors, saving the pickle file
        with it as  save_file_name.pickle.

         If given a pickle file, creates company system based on that

        """

        if logs is None:
            logging.basicConfig(filename="logs/company_system.log", level=logging.ERROR,
                format="%(asctime)s:%(message)s")
        else:
            logger = logging.getLogger(logs)
        self.time_from=datetime.now() - timedelta(days=100)
        self.time_to=time_to
        self.Day =time.strftime("%d-%m-%Y", time.localtime())
        self.Time = time.strftime("%H:%M:%S", time.localtime())
        self.ShortTime = time.strftime("%H%M_%d%m%y", time.localtime())
        if save_file_name is None:
            self.save_file_name = 'backup_pickles/' + self.ShortTime + '.pickle'
        else:
            if date_stamp:
                self.save_file_name = save_file_name + self.ShortTime + '.pickle'
            else:
                self.save_file_name = save_file_name + '.pickle'

        """ This case is when no pickle file is given to load or to upgrade, i.e. to create a new model from scratch.
        """
        if pickle_file is None and pickle_2_update is None:

            self.max_c = max_com

            #     log_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), self.ShortTime + '.log')
            #    logging.basicConfig(filename = 'logs/'+log_name, level=logging.DEBUG)
            #   logging.info('SearchModel initializer run ' + str(self.Day) + " " + str(self.Time))
            self.raw_= []
            self.unique_searches = list(set(raw_texts))
            self.raw_texts=raw_texts
            self.word_vectors = w_vectors
            self.t_l_l = []
            for raw_t in self.raw_texts:
                self.t_l_l.append(text_tools.process_multi(raw_t, self.word_vectors))
            self.item_list = self.create_items(self.unique_searches)
            self.current_file = self.save_file_name
            self.save_up()
        else:
            """Loads an existing model from a file"""
            if pickle_2_update is None:
                with open(pickle_file, 'rb') as f:
                    self.current_file = pickle_file
                    self.item_list = pickle.load(f)
                    self.raw_texts=[]
                    self.t_l_l= []
                    for item in self.item_list:
                        self.t_l_l.append(item.used_tokens)
                        self.raw_texts.append(item.raw_text)
                    self.time_from=self.item_list[0].time_from
                    self.time_to = self.item_list[0].time_to
                    self.word_vectors = w_vectors
                    self.last_update = datetime.fromtimestamp(os.path.getmtime(pickle_file))
            else:
                """Creates a new model, where if new searches are less than max_com, searches from pickle_2_update"""
                print()
                self.time_from = time_from
                self.time_to = time_to
                self.max_c = max_com
                self.raw_texts = raw_texts
                self.unique_searches = list(set(raw_texts))
                self.word_vectors = w_vectors
                self.new_t_l_l = []

                for raw_entry in self.unique_searches:
                    self.new_t_l_l.append(text_tools.process_multi(raw_entry, self.word_vectors))
                with open(pickle_2_update, 'rb') as f:
                    print("items loaded")
                    self.old_item_list = pickle.load(f)
                    self.old_raw_texts = []
                    self.old_t_l_l = []
                    for item in self.old_item_list:

                        self.old_t_l_l.append(item.used_tokens)
                    self.word_vectors = w_vectors
                self.t_l_l= self.new_t_l_l
                self.item_list=self.create_items(self.unique_searches)
                """When less new searches than max_co,m"""
                oldvecs=[x.sum_vec for x in self.old_item_list]
                newvecs=[x.sum_vec for x in self.item_list]

                if  len(self.item_list)< max_com and len(np.unique(oldvecs+newvecs))>max_com:
                    print("HERE")

                    for x in self.old_item_list:
                        if len(self.item_list) < max_com:
                            print(max_com)
                            print(len(self.item_list))
                            print(len(self.old_item_list))
                            print("still here")
                            print(x.used_tokens)
                            if x.used_tokens not in self.t_l_l:
                                print("appending")
                                self.t_l_l.append(x.used_tokens)
                                self.item_list.append(x)
                        else:
                            break
                else:
                    print("IN ELSE LOOP")
                    if len(self.item_list) < max_com:
                        print(max_com)
                        print(len(self.item_list))
                        print(len(self.old_item_list))

                        for x in self.old_item_list:
                            if x.used_tokens not in self.t_l_l:
                                print("adding new")
                                self.t_l_l.append(x.used_tokens)
                                self.item_list.append(x)
                    print("got out")


                self.ShortTime = time.strftime("%H%M_%d%m%y", time.localtime())
                self.improvements = dict()
                self.current_file = self.save_file_name
                self.save_up()

    def create_items(self, raw_texts):
        """Goes through the pairs and description in given raw_pair, creates
        Company objects for those that have a description recognizeable as
        English. Returns list of search items

        """
        working_list = []
        count = 0
        count_ommitted = 0
        counter = 0
        # Pickle the 'data' dictionary using the highest protocol available.
        # for i in range(0, len(self.item_list), 1000):
        #     with open(self.save_file_name + '_' + str(counter) + '.pickle', 'wb') as f:
        #         pickle.dump(self.item_list[i:i + 1000], f, pickle.HIGHEST_PROTOCOL)
        #         counter += 1
        #   logging.info('Saved pickle file into ' + 'pickles/'+self.save_file_name+ '.pickle')
        file_count = 0
        for raw_text in raw_texts:
            if count <= self.max_c - 1:
                if text_tools.is_allowed(raw_text):
                    print(raw_text)
                    # try:
                    #     dec=detect(pair[1])
                    # except:
                    #     dec='error'
                    # if dec=='en':

                    working_list.append(SearchItem.SearchItem(self, raw_text))
                    count += 1
                    # else:
                    #     print("not detected english")
                    if count % 50 == 0:
                        print("At vector no: " + str(
                            count))  # logging.info('Calculated vectors for ' + str(count) + ' entries')  # if count%1000==0:  #     with open(self.save_file_name + '_' +str(file_count)+'.pickle', 'wb') as f:  #         # Pickle the 'data' dictionary using the highest protocol available.  #         pickle.dump(working_list, f, pickle.HIGHEST_PROTOCOL)  #         working_list=[]  #         file_count+=1  #     #   logging.info('Saved pickle file into ' + 'pickles/'+self.save_file_name+ '.pickle')

        return working_list



    def create_or_update(self):
        working_list = []
        count = 0
        count_ommitted = 0
        counter = 0
        # Pickle the 'data' dictionary using the highest protocol available.
        # for i in range(0, len(self.item_list), 1000):
        #     with open(self.save_file_name + '_' + str(counter) + '.pickle', 'wb') as f:
        #         pickle.dump(self.item_list[i:i + 1000], f, pickle.HIGHEST_PROTOCOL)
        #         counter += 1
        #   logging.info('Saved pickle file into ' + 'pickles/'+self.save_file_name+ '.pickle')
        file_count = 0
        for pair in self.texts_and_times:
            if count <= self.max_c - 1:
                if text_tools.is_allowed(pair[0]):
                    # try:
                    #     dec=detect(pair[1])
                    # except:
                    #     dec='error'
                    # if dec=='en':
                    working_list.append(SearchItem.SearchItem(self, pair[0]))
                    count += 1
                    # else:
                    #     print("not detected english")
                    if count % 50 == 0:
                        with open("progress_file.txt" ,"a") as f:

                            f.write("At vector no: " + str(count))  # logging.info('Calculated vectors for ' + str(count) + ' entries')  # if count%1000==0:  #     with open(self.save_file_name + '_' +str(file_count)+'.pickle', 'wb') as f:  #         # Pickle the 'data' dictionary using the highest protocol available.  #         pickle.dump(working_list, f, pickle.HIGHEST_PROTOCOL)  #         working_list=[]  #         file_count+=1  #     #   logging.info('Saved pickle file into ' + 'pickles/'+self.save_file_name+ '.pickle')
        return working_list


    def try_search(self, text_string, display=None, use_wiki=False):
        """returns a Company object with given description and title
        (right now messes with the indices)

        """
        if use_wiki:
            new_string = text_tools.get_wiki_categories(text_string, self.word_vectors)
            if new_string==text_string:
                used_wiki=False
            else:
                new_string = new_string.replace("American", "")
                used_wiki=True
        else:
            used_wiki=False
            if display==None:
                new_string = text_string
            else:
                new_string=text_string
                text_string=display
        return SearchItem.SearchItem(self, raw_text=new_string, display_text=text_string, used_wiki=used_wiki)


    def save_up(self):
        """With given file name, saves the pickled search item system
        """
        with open(self.save_file_name, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.item_list, f,
                pickle.HIGHEST_PROTOCOL)  # logging.info('Saved pickle file into ' + 'pickles/'+self.save_file_name+ '.pickle')

    def similar_by_SearchItem(self, search_item, cosine=True, print_list=True, top_n=5):
        """Returns the top_n similar search items(closest sum vectors) to the
        given search item
        """

        close_search_items = text_tools.knn_search(search_item=search_item, D=self.item_list,
                cosine=cosine, K=top_n)
        if print_list:
            text_tools.print_comparison(self, search_item=search_item, compare_items=close_search_items, cosine=cosine,
                top_n=top_n)
            print('----------')
        return close_search_items

    def random_similar(self, cosine=True, print_list=True, top_n=5):
        """returns  a dictionary object where the keys are top_n random
        search items, with the value being pairs of the 5 closest search items
        """

        shuffled = self.item_list.copy()
        np.random.shuffle(shuffled)
        np.random.seed()

        similar_dict = dict()
        for search_item in shuffled[:top_n + 1]:
            close_search_items = self.similar_by_SearchItem(search_item, self.item_list, print_list, top_n)
            similar_dict.update({search_item: (i[0] for i in close_search_items[:top_n])})
        return similar_dict



    def similar_by_text(self, search_string, print_list=False,  use_wiki=True, cosine=True, top_n=7 ,max_l=1000, count_c='c'):
        """"Returns the top_n most similar search items to given text string"""
        if self.in_model(search_string):
            matched_model=self.get_item_with_text(search_string)
            seen = [matched_model]
            close_unique = [matched_model]
        else:
            seen = []
            close_unique = []
        search_item = self.try_search(search_string, use_wiki=use_wiki)
        close_search_items = self.knn_with_criteria(search_item=search_item, D=self.item_list,
            cosine=cosine,
            K=top_n * 3,
            max_lenght=max_l, count_char=count_c)



        for comp in close_search_items:
            if comp.display_text.lower() not in seen:
                seen.append(comp.display_text.lower())
                close_unique.append(comp)
        if print_list:
            text_tools.print_comparison(self, search_item=search_item, compare_items=close_unique, cosine=cosine,
                top_n=top_n)
        return close_unique[:top_n]

    def similar_by_vector(self, comparison_vector, print_list=True, cosine=True, top_n=5):
        """returns a list of top_n similar search items for given vector"""

        close_search_items = text_tools.knn_search_vector(vector=comparison_vector, D=self.item_list, cosine=cosine,
            K=top_n*3)
        seen = []
        close_unique = []
        for comp in close_search_items:
            if comp.display_text.lower() not in seen:
                seen.append(comp.display_text.lower())
                close_unique.append(comp)
            else:
                print(comp.display_text)
                print(comp.raw_text)
        if print_list:
            text_tools.print_comparison_vec(self, search_vec=comparison_vector, compare_items=close_unique,
                cosine=cosine, top_n=top_n)
        return close_unique[:top_n]


    def closest_parts(self, search_item_1, search_item_2, cosine=True, close_words_n=4, one_side=True):
        """returns a list of the  word combinations of at most close_words_n in search_item_2.tokens
            that is closest to the sum_vector of given search_item_1
            """

        if one_side:
            ngrams = []
            for n in range(close_words_n):
                ngrams += list(ng(search_item_2.used_tokens, n + 1))
            #            for token in search_item.used_tokens:
            #                ngrams+=tuple(token)
            print(ngrams)
            weights = []
            return_dict = dict()

            for ngram in ngrams:
                sum_vector = np.zeros(self.word_vectors.vector_size)
                for token in ngram:
                    if token in search_item_2.used_tokens:
                        count_token = search_item_2.used_tokens.count(token)
                        sum_vector = sum_vector + np.multiply(
                            text_tools.tfidf(token, search_item_2.lower_tokens, self.t_l_l) * self.word_vectors.wv[token],
                            1 / count_token)
                weights.append(sum_vector)

                return_dict.update({ngram: text_tools.similarity_vec(sum_vector, search_item_1.sum_vec, cosine)})

            sd = list(sorted(return_dict.items(), key=operator.itemgetter(1)))
            for ppp in sd:
                print(ppp)
            if cosine:
                sd = sd[::-1]
            return sd[0]
        else:
            return self.closest_parts(search_item_1, search_item_2, cosine, close_words_n, one_side=True), self.closest_parts(
                search_item_2, search_item_1, cosine, close_words_n, one_side=True)

    def closest_parts_vector(self, vector, search_item, cosine=True, close_words_n=3):
        """returns the word combination of at most close_words_n in search_item.lower_tokens
        that is closest to the sum_vector of given search item
        """

        ngrams = []
        for n in range(close_words_n):
            ngrams += list(ng(search_item.used_tokens, n + 1))
        #            for token in search_item.used_tokens:
        #                ngrams+=tuple(token)
        print(ngrams)
        weights = []
        return_dict = dict()

        for ngram in ngrams:
            sum_vector = np.zeros(self.word_vectors.vector_size)
            for token in ngram:
                if token in search_item.used_tokens:
                    count_token = search_item.used_tokens.count(token)
                    sum_vector = sum_vector + np.multiply(
                        text_tools.tfidf(token, search_item.lower_tokens, self.t_l_l) * self.word_vectors.wv[token],
                        1 / count_token)
            weights.append(sum_vector)

            return_dict.update({ngram: text_tools.similarity_vec(sum_vector, vector, cosine)})

        sd = list(sorted(return_dict.items(), key=operator.itemgetter(1)))
        for ppp in sd:
            print(ppp)
        if cosine:
            sd = sd[::-1]
        return sd[0]

    def search_item_by_index(self, given_index):
        """return the search item in system with given index
            """
        return self.item_list[given_index]

    def main_terms(self, search_item_1, top_n=5):
        """Returns top_n tokens from the given search item with the highest weights """
        sorted_by_weights = sorted(search_item_1.token_weights.items(), key=operator.itemgetter(1))
        return sorted_by_weights[-top_n:]

    def similar_search_item_1_2(self, search_item_1_int, search_items, comparison_ratio=0.5, top_n=5):
        """Given search item by index search item_1_int to compare with set search items of
        indexes, gives results more similar to search items by comparison_factor,
        where =0 is none, =1 fullt similar to search items"""
        search_item_1 = self.search_item_by_index(search_item_1_int)
        if isinstance(search_items, int):
            average_vector = self.search_item_by_index(search_items).sum_vec
        else:
            average_vector = np.mean([self.search_item_by_index(comp_idx).sum_vec for comp_idx in search_items], axis=0)
        comparison_vector = average_vector - search_item_1.sum_vec
        self.similar_by_vector(search_item_1.sum_vec + comparison_vector * comparison_ratio)

    def search_tags(self, tokens, comparison_factor=1, search_item_idx=None, top_n=5):
        all_tags = ''
        for x in range(comparison_factor):
            all_tags += tokens
        if search_item_idx is None:
            search_item = self.try_search(all_tags)
            close_search_items = [self.item_list[x] for x in text_tools.knn_search(search_item, self.item_list, 30)]

            print('Given tags ' + tokens + 'are close to: ')
            search_items_distances = []

            for close_search_item in close_search_items:
                if not close_search_item.display_text == search_item.display_text:
                    search_items_distances.append(
                        [close_search_item, sqrt(((search_item.sum_vec - close_search_item.sum_vec) ** 2).sum(axis=0))])
            for comp_and_dist in search_items_distances[:top_n + 1]:
                print('Distance: ' + str(comp_and_dist[1]) + ' ' + comp_and_dist[0].display_text + '[' + str(
                    self.item_list.index(comp_and_dist[0])) + ']' + '\n' + comp_and_dist[0].raw_desc)
            print('-------------')
        else:
            search_item_1 = self.search_item_by_index(search_item_idx)
            search_item = self.try_search(search_item_1.raw_desc + all_tags)
            close_search_items = [self.item_list[x] for x in text_tools.knn_search(search_item, self.item_list, 30)]

            print('Company: \"' + search_item_1.display_text.upper() + ' [' + str(
                self.item_list.index(search_item_1)) + ']' + 'with added tags ' + tokens + " x" + str(
                comparison_factor) + ' is close to: ')
            search_items_distances = []

            for close_search_item in close_search_items:
                if not close_search_item.display_text == search_item.display_text:
                    search_items_distances.append(
                        [close_search_item, sqrt(((search_item.sum_vec - close_search_item.sum_vec) ** 2).sum(axis=0))])
            for comp_and_dist in search_items_distances[:top_n + 1]:
                print('Distance: ' + str(comp_and_dist[1]) + ' ' + comp_and_dist[0].display_text + '[' + str(
                    self.item_list.index(comp_and_dist[0])) + ']' + '\n' + comp_and_dist[0].raw_desc)
            print('-------------')

    def find_title(self, title):
        """return search item or search items with given string in its name"""
        for search_item in self.item_list:
            if title.lower() in search_item.raw_name.lower():
                return search_item
        else:
            print("No search item found! \n")
            return self.item_list[0]

    def get_index(self, search_item):
        """given a search item object, returns index"""
        return self.item_list.index(search_item)

    def similar_by_index(self, idx, cosine__=True, top_n__=5):
        """Returns the top_n__ most similar entries to self.item_list[idx]"""

        return self.similar_by_SearchItem(search_item=self.search_item_by_index(idx), cosine=cosine__, top_n=top_n__)

    def similar_two_text(self, text_1, text_2, cosine=True):
        """Returns the similarity score or distance"""

        search_item_1 = Company.Company(self, 'Example search_item 1', text_1)
        search_item_2 = Company.Company(self, 'Example search_item 1', text_2)
        return text_tools.similarity(search_item_1, search_item_2, cosine)

    def furthest_vector(self, vectors, cosine=True):
        """Returns the vector least similar to the mean of the given vectors"""

        average_vector = np.mean(vectors, axis=0)
        furthest = [self.item_list[x] for x in
            text_tools.knn_s_all_vector(average_vector, vectors, cosine, furthest=True, k=1)]
        return furthest[0]

    def furthest_search_items(self, search_items, cosine=True):

        average_vector = np.mean([i.sum_vec for i in search_items], axis=0)
        furthest = [text_tools.knn_search_vector(average_vector, search_items, cosine, furthest=True, k=1)]
        return furthest[0]

    def move_out(self, source_search_item, move_search_item, outside_set, cosine=None):
        """moves the given vector in the direction of MINUS so that the similarity
        distance between source_search_item and move_search_item is the same as the biggest
        distance in the given set"""
        MINUS = move_search_item.sum_vec - source_search_item.sum_vec
        MIN_RADIUS = max([np.linalg.norm(source_search_item.sum_vec - i.sum_vec) for i in outside_set])
        move_search_item.sum_vec = MIN_RADIUS * MINUS / np.linalg.norm(MINUS)
        move_search_item.modified = True

    # logging.info('Company ' + str(move_search_item.index) + " moved outside of suggestion "
    #                     + str([c.index for c in outside_set])+ ' of ' + str(move_search_item.index))

    def switch_rank(self, isource_search_item, imove1, imove2):
        """switches the similarity ranks of search items of index imove1 and imove2
        in relation to their similarity to search_item index isource_search_item"""
        source_search_item = self.search_item_by_index(isource_search_item)
        move1 = self.search_item_by_index(imove1)
        move2 = self.search_item_by_index(imove2)
        MINUS1 = move1.sum_vec - source_search_item.sum_vec
        MINUS2 = move2.sum_vec - source_search_item.sum_vec
        dist1 = np.linalg.norm(source_search_item.sum_vec - move1.sum_vec)
        dist2 = np.linalg.norm(source_search_item.sum_vec - move2.sum_vec)
        move1.sum_vec = source_search_item.sum_vec + dist2 * MINUS1 / np.linalg.norm(MINUS1)
        move1.modified = True
        move2.sum_vec = source_search_item.sum_vec + dist1 * MINUS2 / np.linalg.norm(MINUS2)
        move2.modified = True

    #        MOVE=move_search_item.sum_vec
    #        MOVE_VEC=move_search_item.sum_vec-source_search_item.sum_vec
    #        SOURCE=source_search_item.sum_vec
    #        NORM=np.linalg.norm(MOVE)
    #        MIN_RADIUS=max([np.linalg.norm(source_search_item.sum_vec-i.sum_vec) for i in outside_set])
    #        print(MIN_RADIUS)
    #        PROJ_VEC= text_tools.project_vector(MOVE, SOURCE)
    #        print('proj')
    #        print(PROJ_VEC)
    #        MID_VEC=sqrt((np.linalg.norm(SOURCE)**2 - np.linalg.norm(PROJ_VEC)**2))
    #        print('mid')
    #        print(MID_VEC)
    #        RAY_VEC=sqrt((MIN_RADIUS**2 - MID_VEC**2))
    #        print('ray')
    #        print(RAY_VEC)
    #        return MOVE*1/NORM*(PROJ_VEC+RAY_VEC)
    #
    #

    def print_all(self, cosine__=None, top_n__=5):
        """Prints top_n__ similar entries to every item in the model"""
        for search_item in self.item_list:
            self.similar_by_SearchItem(search_item, cosine=cosine__, top_n=top_n__)

    def is_newest_updateable(self):
        """Checks if model is based on the newest pickle file """

        return not self.newest_model() == self.current_file and text_tools.good_pickle(self.newest_model())

    def change_pickle(self, change_to_this):
        """changes the pickle file of current SearchModel to """
        with open(change_to_this, 'rb') as f:
            self.current_file = change_to_this
            self.item_list = pickle.load(f)
            self.raw_texts=[]
            self.t_l_l= []
            for item in self.item_list:
                self.t_l_l.append(item.used_tokens)
                self.raw_texts.append(item.raw_text)
            self.time_from=self.item_list[0].time_from
            self.time_to = self.item_list[0].time_to
            self.last_update = datetime.fromtimestamp(os.path.getmtime(change_to_this))

    def should_update(self, time_interval):
        """Returns True or False, dependent on whethher time_interval minutes has passed since last update/"""
        if (datetime.now() - self.last_update) > timedelta(minutes=time_interval):
            return True
        else:
            return False

    def full_update(self, interval):
        """If the time in minutes since interval has elapsed since last update, updates pickle file is a valid update
        exits and returns True. Else returns false."""
        if self.should_update(interval):
            newest = self.newest_model()
            if text_tools.good_pickle(newest):
                self.change_pickle(newest)
                return True
            else:
                return False


    def knn_with_criteria(self, search_item, D, cosine=False, furthest=False, K=5, max_lenght=1000, count_char='c'):
        """ gives the indices of the K nearest vectors to the sum vector of data among D """
        # euclidean distances from the other points
        if cosine:
            sqd = text_tools.flat_list(text_tools.flat_list(
                [text_tools.cosine_similarity(np.asmatrix(search_item.sum_vec), np.asmatrix(y)) for y in [z.sum_vec for z
                    in D]]))
            idx = argsort(sqd)
        else:
            sqd = [sqrt(((y - (search_item.sum_vec)) ** 2).sum(axis=0)) for y in [z.sum_vec for z in D]]
            idx = argsort(sqd)[::-1]  # sorting
            # return the indexes of K nearest neighbours
        if furthest:
            idx = idx[::-1]
        return_list = []
        for idn in idx:
            if count_char=='c':
                if len(self.search_item_by_index(idn).display_text)<=max_lenght:
                    return_list.append(D[idn])
            else:
                if len([x for x in word_tokenize(self.search_item_by_index(idn).display_text) if x.isalpha()])<=max_lenght:
                    return_list.append(D[idn])
        if search_item in return_list:
            return_list.remove(search_item)
        return return_list[::-1][:K]


    def newest_model(self):
        """Returns newest built pickle"""
        files = glob.glob('pulled_pickles/*')
        if len(files) > 0:
            return (max(files, key=os.path.getctime))
        else:
            return ('backup_pickles/gensim_freq.pickle')

    def join_item_lists(self, list1, list2):
        new_list=[]
        used_from_2=[]
        for item in list1:
            if item.raw_text in [x.raw_text for x in list2]:
                new_list.append(item.add_times(self.get_item_by_text(item.raw_text, list2).times))
                used_from_2.append(self.get_item_by_text(item.raw_text, list2))
            else:
                new_list.append(item)
        for item in list2:
            if item not in used_from_2:
                new_list.append(item)
        return new_list

    def get_item_by_text(self, text, list):
        result= [x for x in list if x.raw_text==text]
        if not result==[]:
            return result[0]
        else:
            return None

    def remove_until_this_many(self, this_many):
        while len(self.item_list)>this_many:
            for search_event in self.texts_and_times:
                item=self.get_item_by_text(search_event[0])
                if self.get_item_by_text(search_event[0]).remove_time(search_event[1]):
                    continue
                else:
                    (self.item_list).remove(self.get_item_by_text(search_event[0]))

    def change_timeframe(self, time_from, time_to):
        self.time_from=time_from
        self.time_to=time_to
        for item in self.item_list:
            item.change_timeframe(time_from, time_to)

    def in_model(self, text):
        processed= [x.lower() for x in word_tokenize(text)]
        return processed in [x.lower_tokens for x in self.item_list]

    def get_item_with_text(self, text):
        processed = [x.lower() for x in word_tokenize(text)]
        for item in self.item_list:
            if item.lower_tokens == processed:
                return item
        return self.item_list[0]
