import os
import unittest

from test.decision_tree import DecisionTreeTestCase
from test.svm import SVMTestCase
from test.gaussian_process import GaussianProcessTestCase
from test.naive_bayes import NaiveBayesTestCase
from test.nearest_centroid import NearestCentroidTestCase
from test.sgd import SGDTestCase
from test.lstm import LSTMTestCase
from test.text_cnn import TextCNNTestCase
from test.transformer import TransformerTestCase


if __name__ == '__main__':
    unittest.main()
    os.remove(DecisionTreeTestCase.SAVED_PATH)
