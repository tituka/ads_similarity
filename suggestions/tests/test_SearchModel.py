import random
import unittest
import sys
import os
print( os.getcwd())
sys.path.insert(0, "model_components")

import SearchModel
import numpy as np
from gensim.models import KeyedVectors
sys.path.insert(0, "tools")
print( os.getcwd())

from text_tools import similarity, similarity_vec

word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
cwd = os.getcwd()
print(cwd)
model = SearchModel.SearchModel(w_vectors=word_vectors, pickle_file='backup_pickles/glove_backup.pickle')


def find_text(text):
    """return search_item or companies with given string in its name"""
    for search_item in model.item_list:
        if text.lower() in search_item.display_text.lower():
            return search_item
    else:
        print("No search_item found! \n")
        return model.item_list[0]


def random_search_item(ind):
    shuffled = model.item_list
    np.random.shuffle(shuffled)
    np.random.seed()
    return shuffled[ind]


class TestSearchModel(unittest.TestCase):
    def test_try_search_item(self):
        search_text = 'hee ae some words and more oblique and obtuse onES iwth /// ands whatever'
        test_search_item1 = model.try_search(search_text)
        test_search_item2 = model.try_search(search_text + "test words")
        self.assertEqual(test_search_item1.display_text, search_text)


        tested1 = np.array_equal(test_search_item1.sum_vec, test_search_item2.sum_vec)

        print(test_search_item1.display_text)
        print(test_search_item1.sum_vec)
        print(test_search_item2.sum_vec)
        print(test_search_item1.display_text)

        print(tested1)
        self.assertFalse(tested1)


    def test_similar_by_text(self):
        int = random.randint(0, len(model.item_list))
        item_1 = random_search_item(int)
        counter=1
        while item_1.sum_vec.all()==np.zeros(model.word_vectors.vector_size).all():
            item_1 = random_search_item(counter)
            counter += 1
        item_2 = random_search_item(counter)
        while np.array_equal(item_1.sum_vec, item_2.sum_vec) or np.array_equal(item_2.sum_vec, np.zeros(
                model.word_vectors.vector_size)):
            counter += 1
            item_2 = random_search_item(counter)
        res1 = model.similar_by_text(item_1.display_text, cosine=True, top_n=5)
        res2 = model.similar_by_text(item_2.display_text, cosine=False, top_n=5)
        res3 = model.similar_by_text(item_2.display_text, cosine=True, top_n=5, print_list=True)
        """Assets that two different random search items do mot return the same similar searches """
        self.assertNotEqual(res3, res1)
        """Asserts that returned similar searches are more similar, the higher in the ranking they are"""
        self.assertTrue(similarity(item_1, res1[0]) > similarity(item_1, res1[3]))
        self.assertTrue(similarity(item_2, res2[0], cosine=False) < similarity(item_2, res2[3], cosine=False))


    def test_similar_by_search_item(self):
        int= random.randint(0, len(model.item_list))
        test_search_item1 = random_search_item(int)
        res1 = model.similar_by_SearchItem(test_search_item1, cosine=True, top_n=10)
        res2 = model.similar_by_SearchItem(test_search_item1, top_n=10)
        print(res1)
        print(res2)
        print(similarity(test_search_item1, res1[0]))
        print(similarity(test_search_item1, res1[3]))
        print(similarity(test_search_item1, res2[0], cosine=False))
        print(similarity(test_search_item1, res2[3], cosine=False))
        self.assertTrue(similarity(test_search_item1, res1[0]) > similarity(test_search_item1, res1[3]))
        self.assertTrue(similarity(test_search_item1, res2[0], cosine=False) < similarity(test_search_item1, res2[3], cosine=False))


    def test_similar_by_vector(self):
        ind = random.randint(0, len(model.item_list)-1)
        test_search_item1 = random_search_item(ind)
        res2 = model.similar_by_vector(test_search_item1.sum_vec, cosine=False)
        self.assertTrue(similarity(test_search_item1, res2[0], cosine=False) < similarity(test_search_item1, res2[3], cosine=False))


    def test_similar_by_vector_C(self):
        ind = random.randint(0, len(model.item_list)-1)
        test_search_item1 = random_search_item(ind)
        res1 = model.similar_by_vector(test_search_item1.sum_vec, cosine=True)
        self.assertTrue(similarity(test_search_item1, res1[0], cosine=True) > similarity(test_search_item1, res1[3], cosine=True))
       

    def test_move_out(self):
        ind = random.randint(0, len(model.item_list)-1)

        item = random_search_item(ind)
        similars = model.similar_by_SearchItem(item)
        old_position = similars[2].sum_vec
        move_vec = similars[2]
        model.move_out(item, similars[2], similars, cosine=False)
        self.assertTrue(similarity(item, move_vec) < similarity_vec(item.sum_vec, old_position))
        similars = model.similar_by_SearchItem(item)
        self.assertNotEqual(move_vec, similars[2])
        self.assertEqual(move_vec.modified, True)
        move_vec.reset()
        self.assertEqual(move_vec.modified, False)

    def test_in_model(self):
        ind = random.randint(0, len(model.item_list) - 1)
        item = random_search_item(ind)
        self.assertTrue(model.in_model(item.display_text))
        self.assertFalse(model.in_model("this text certainly won't be in the model"))

    def test_get_item_with_text(self):
        ind = random.randint(0, len(model.item_list) - 1)
        item_1 = random_search_item(ind)
        item_2 = random_search_item(ind)
        counter=1
        while item_1.sum_vec.all() == np.zeros(model.word_vectors.vector_size).all():
            item_1 = random_search_item(counter)
            counter += 1

        while item_2.sum_vec.all() == np.zeros(model.word_vectors.vector_size).all() or item_1==item_2:
            item_2 = random_search_item(counter)
            counter += 1
        self.assertEqual(item_1, model.get_item_with_text(item_1.display_text.upper()))
        self.assertNotEqual(model.get_item_with_text(item_1.display_text.upper()), model.get_item_with_text(
            item_2.display_text.upper()))


        """
#    def test_closest_parts(self):
#        comp1=model.try_search_item('dogs cats pets animals pets')
#        comp2=model.try_search_item('desk chair meeting dog')
#        parts=model.closest_parts(comp1, comp2, cosine=False)
#        parts2=model.closest_parts(comp1, comp2, cosine=True)
#        self.assertEqual(parts, ['dog'])
#   self.assertEqual(parts2, ['dog'])


#        
#    def test_closest_parts_vector(self):
#
#        comp1=model.try_search_item('MOCO is a smart and lean cloud software made for small medium-sized agency and service businesses.')
#        comp2=model.try_search_item('MascotaSocial - The socialPet network for pet lovers and pet service`s owners. For making friends, protect, adopt, enjoy and explore!"')
#        
#
#        parts=model.closest_parts_vector(comp1.sum_vec, comp2, cosine=True)
#        print(parts)
#     #   parts2=model.closest_parts_vector(comp2.sum_vec, comp1, cosine=True)
#   #     self.assertEqual(parts, ['dog'])
#   #     self.assertEqual(parts2, ['dog'])   


#    def test_similar_two_text(self):
#        desc1='MOCO is a smart and lean cloud software made for small medium-sized agency and service businesses.'
#        desc2= "BankBazaar.com is the world's first neutral online marketplace that gives you instant customized rate quotes on loans and insurance products. You can instantly search for, compare and apply for loans, credit cards and insurance products on our site. Since we partner with India's leading financial institutions and insurance firms, you have to look in only one place to get a great deal."
#        score1=model.similar_two_text(desc1, desc2, cosine=True)
#        score2=model.similar_two_text(desc2, desc2, cosine=True)
#        score21=model.similar_two_text(desc1, desc2, cosine=False)
#        score22=model.similar_two_text(desc2, desc2, cosine=False)
#        self.assertEqual(round(score2, 2), 1)
#        self.assertNotEqual(score1, 0)
#        self.assertEqual(round(score22, 2), 0)
#        self.assertNotEqual(score21, 0)
#     

#    def test_furthest_vector(self):
#        desc1="Moofio is a brand new social network for pets and pet lovers, that is available on App Store for free.a newsfeed based on your friends and specific sections for your needs as a pet. Just like your favourite social network. But for pets. Moofio is so much more fun this way. Second option (A great option for â€œnot yet have a petâ€ pet lovers) sign up as a pet lover. Functionality, the same. But if you have a pet, better to sign up as it. Why being an human in a pet social network? Nobody likes a killjoy. From here on in, you may find helping hands from Moofioâ€™s kind community for your petâ€™s needs or just browse around adorable puppies, smart-ass cats and all the lovely creatures of mother nature." 
#        desc2="Coinbase is a digital wallet that allows you to securely buy, use, and accept bitcoin currency.n online platform that allows merchants, consumers, and traders to transact with digital currency. It allows its users to create their own bitcoin wallets and start buying or selling bitcoins by connecting with their bank accounts. In addition, it provides a series of merchant payment processing models and tools that support many highly-trafficked websites on the internet. Coinbase mission is to create an open financial model for the world. It is operated from San Francisco, California."
#        idx11=847
#        idx1=806
#        idx2=1751
#
#        ret1=model.similar_by_index(100, cosine=True, top_n=7)
#        close=model.item_list[774]
#        print(close.display_text)
#        print("JJJJJJJJJJJJJJJJJJJJJJJJJ")
#        comps=[i[0] for i in ret1]
#        for comp in comps:
#            print(comp.display_text)
#        print("HEREITIS")
#        comps+=[close] 
#        res=model.furthest_companies(comps)[0]
#        model.print_comparison(comps[0], comps)

#        for j in [i[0] for i in ret1]:
#            print(similarity(res, j))
#        other=[i[0] for i in ret1][2]
#        for j in [i[0] for i in ret1]:
#            print(similarity(other, j))
#        
"""

if __name__ == '__main__':
    unittest.main()
