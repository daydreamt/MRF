import unittest
import torch

from mrf.mrf import MRF

class MRFTest1(unittest.TestCase):
    def test_works(self):
        words = ['hello', 'this', 'text', 'is', 'out', 'of', 'BANANA']
        priors = [0.3,   0.5,           0.3,    0.5,       0.5,    0.5,         0.9]
        mrf2 = MRF(words, priors, torch.FloatTensor([[0.6, 0.4], [0.4, 0.6]]))
        mrf2.make_inference_and_get_beliefs()
        self.assertTrue(True) # We reached this far



class MRFTest1(unittest.TestCase):
    def test_works(self):
        words = ['hello', 'this', 'text', 'is', 'out', 'of', 'BANANA']
        priors = [0.5,   0.5,           0.5,    0.5,       0.5,    0.5,         0.7]
        print(list(zip(words, priors)))
        mrf1 = MRF(words, priors, torch.FloatTensor([[0.9, 0.1], [0.1, 0.9]]))

        print(mrf1.get_univariate_potential_function(6)(1))
        print(mrf1.get_univariate_potential_array(0))
        print(mrf1.get_pairwise_potential_function(1,2)(0, 1))
        print(mrf1.get_initial_messages())
        print(mrf1.set_initial_messages())
        mrf1.get_belief(0)
        some_messages = mrf1.get_initial_messages()
        message1 = mrf1.get_message(0, 1, some_messages)
        print("*" * 80)
        message1
        mrf1.make_inference()
        beliefs = mrf1.make_inference_and_get_beliefs()
        self.assertTrue(True) # We reached this far


if __name__ == '__main__':
    unittest.main()