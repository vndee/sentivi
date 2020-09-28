from sentivi.base_model import ClassifierLayer
from .naive_bayes import NaiveBayesClassifier
from .svm import SVMClassifier
from .mlp import MLPClassifier
from .decision_tree import DecisionTreeClassifier
from .sgd import SGDClassifier
from .nearest_centroid import NearestCentroidClassifier
from .gaussian_process import GaussianProcessClassifier
from .text_cnn import TextCNNClassifier
from .lstm import LSTMClassifier


class RNNClassifier(ClassifierLayer):
    def __init__(self):
        super(RNNClassifier, self).__init__()


class TransformerClassifier(ClassifierLayer):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
