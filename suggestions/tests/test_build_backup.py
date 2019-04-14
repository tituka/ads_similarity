import random
import unittest
from datetime import timedelta, datetime
from gensim.models import KeyedVectors
import os
import sys
sys.path.insert(0, "model_components")
import SearchModel
sys.path.insert(0, "build_update")
import build_backup

def roundup(x):
    """A simple function to round up to the nearest 5000"""
    return x if x % 1000 == 0 else x + 1000 - x % 1000


word_vectors = KeyedVectors.load_word2vec_format("vectors/gensim_glove_vectors25.txt", binary=False)
time_to = datetime.now()
time_from = datetime.now() - timedelta(days=100)


class Test_build_backup(unittest.TestCase):

    def test_get_searches(self):
        rng = random.SystemRandom()
        now=datetime.now()
        earlier=rng.randint(1,10000)
        random_total=rng.randint(1,15000)
        searches= build_backup.get_searches(random_total, time_from, time_to)
        print("RANDOMHERE")
        print(random_total)
        print(round(14555, -3))

        """Tests that get_searches returns the right number of searches"""
        if random_total<1000:
            self.assertTrue(len(searches)< random_total)
        else:
            """Because of pagination, some rounding up is needed"""
            self.assertTrue(len(searches)<roundup(random_total))

        """Tests that returned searches are in desired timeframe"""
        test_model=SearchModel.SearchModel( raw_texts=searches[:10], w_vectors=word_vectors,
            save_file_name="DELETE_THIS",  date_stamp=False, time_from=earlier, time_to=now)
        for item in test_model.item_list:
            print(item.this_many)
            self.assertGreater(item.this_many, 0, "No searches for term" + item.display_text + " found in timefrane")

        """Removes the test model that was used for previous step"""
        os.remove("DELETE_THIS.pickle")


if __name__ == '__main__':
    unittest.main()