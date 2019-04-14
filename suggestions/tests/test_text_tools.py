import random
import unittest
from nltk import word_tokenize
from gensim.models import KeyedVectors
import numpy as np
import os
import sys
sys.path.insert(0, "model_components")
import SearchModel
sys.path.insert(0, "tools")
import text_tools
word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
model =SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/glove_backup.pickle')


def find_title(title):
    """return company or companies with given string in its name"""
    for item in model.item_list:
        if title.lower() in item.display_title.lower():
            return item
    else:
        print("No company found! \n")
        return item.item_list[0]


def random_search_item(ind):
    shuffled = model.item_list
    np.random.shuffle(shuffled)
    np.random.seed()
    return shuffled[ind]


class Test_text_tools(unittest.TestCase):

    def test_similarity_vec(self):
        int = random.randint(0, len(model.item_list))
        item_1 = random_search_item(int)
        counter = 1
        while item_1.sum_vec.all() == np.zeros(model.word_vectors.vector_size).all():
            item_1 = random_search_item(counter)
            counter += 1
        item_2 = random_search_item(counter)
        while np.array_equal(item_1.sum_vec, item_2.sum_vec) or np.array_equal(item_2.sum_vec,
                np.zeros(model.word_vectors.vector_size)):
            counter += 1
            item_2 = random_search_item(counter)
        sim = text_tools.similarity_vec(item_1.sum_vec, item_2.sum_vec)
        sim2 = text_tools.similarity_vec(item_2.sum_vec, item_1.sum_vec)
        self.assertEqual(sim, sim2)

    def test_find_compound(self):
        """Unit test for finding compounds starting at gio"""
        litania = ["There", "is", "a", "thing", "in", "the", "k", "bot", "model"]
        best1 = text_tools.find_compound("is", 1, litania, word_vectors)
        best2 = text_tools.find_compound("k", 6, litania, word_vectors)
        self.assertEqual(best1, ("is", 0))
        self.assertEqual(best2, ("k_bot", 1))

    def test_replace_with_multi(self):
        print("--------------------------------------------")
        print("is" in word_vectors)
        litania = ["there", "is", "a", "thing", "in", "the", "k", "bot", "model"]
        self.assertEqual(text_tools.replace_with_multi(litania, word_vectors),
            ["there", "is", "a", "thing", "in", "the", "k_bot", "model"])

    def test_all_concat(self):
        litania1 = "yy kaK koo nee VII KUU"
        self.assertEqual(text_tools.all_concat(word_tokenize(litania1)), [['yy', 0], ['yy_kaK', 1], ['yy_kaK_koo', 2],
            ['yy_kaK_koo_nee', 3], ['yy_kaK_koo_nee_VII', 4], ['yy_kaK_koo_nee_VII_KUU', 5]])

    def test_process_multi(self):
        litania1 = text_tools.process_multi("one two three for k bot five six whatever", word_vectors)
        litania2 = text_tools.process_multi("K boT one two three for five k bot.", word_vectors)
        self.assertEqual(litania1, ['one', 'two', 'three', 'for', 'k_bot', 'five', 'six', 'whatever'])
        self.assertEqual(litania2, ['k_bot', 'one', 'two', 'three', 'for', 'five', 'k_bot', '.'])

    def test_are_repetitons(self):
        litania1 = "yy kaa koo nee vii"
        litania2 = "yyy tt kaa yyy koo  yyy"
        litania3 = "yy kaa kolme neljä kolme kolme"
        self.assertFalse(text_tools.are_repetitions(litania1))
        self.assertTrue(text_tools.are_repetitions(litania3))
        self.assertFalse(text_tools.are_repetitions(litania2))

    def test_missing_in_model(self):
        litania1="k bot"
        litania2="some other shit"
        self.assertTrue(text_tools.missing_in_model(litania1, 3, word_vectors))

    def test_get_wiki_categories(self):
        litania1= "Blaablaablaa jotainjotain"
        litania2="James Deen"
        litania3="Tiina"
        self.assertEqual(text_tools.get_wiki_categories(litania1, word_vectors),litania1 )
        self.assertTrue("American male actors" in text_tools.get_wiki_categories(litania2, word_vectors))
        self.assertEqual(text_tools.get_wiki_categories(litania3, word_vectors), litania3)

    def test_process_for_display(self):
        litania= model.try_search("Here are some words to capitalize")
        self.assertEqual(text_tools.process_for_display(litania.lower_tokens),"Here Are Some Words to Capitalize")


if __name__ == '__main__':
    unittest.main()
