import random
import unittest
from gensim.models import KeyedVectors
import sys
import os
print( os.getcwd())

sys.path.insert(0, "model_components")
import SearchItem, SearchModel

print( os.getcwd())

word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
model = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/backup.pickle')
random.seed(3)


def random_search():
    return model.search_item_by_index(random.randint(0, len(model.item_list)-1))


class Test_SearchItem(unittest.TestCase):

    def test_SearchItem(self):
        """Test basic company creation, storing display text et"""
        search_text = 'Description unique words test and stuff\n'
        item1 = SearchItem.SearchItem(model, search_text)
        self.assertEqual(item1.raw_text, search_text)
        self.assertEqual(item1.display_text, 'Description unique words test and stuff')

    def test_get_vector(self):
        "test tokenization, that capitals are handled properly"
        text1 = 'Description unique words test and stuff\n'
        text2 = 'DescrIption unique words test and stuff\n'
        text3 = 'DescrIption words test stuff\n and unique'
        text4 = 'Description extra unique words test surprise and stuff\n'
        comp1 = SearchItem.SearchItem(model, text1)
        comp2 = SearchItem.SearchItem(model, text2)
        comp3 = SearchItem.SearchItem(model,  text3)
        comp4 = SearchItem.SearchItem(model, text4)
        self.assertEqual(set(comp1.used_tokens), set(comp2.used_tokens))
        self.assertEqual([round(comp1.weights_sum, 3), set(comp1.token_weights)],
            [round(comp3.weights_sum, 3), set(comp3.token_weights)])
        self.assertNotEqual(set(comp1.used_tokens), set(comp4.used_tokens))

    def test_reset(self):
        comp = random_search()
        comp.reset()
        self.assertEqual(comp.modified, False)


if __name__ == '__main__':
    unittest.main()