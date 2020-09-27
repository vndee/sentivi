from sentivi.base_model import ClassifierLayer
from .naive_bayes import NaiveBayesClassifier
from .svm import SVMClassifier
from .mlp import MLPClassifier
from .decision_tree import DecisionTreeClassifier


class CNNClassifier(ClassifierLayer):
    def __init__(self):
        super(CNNClassifier, self).__init__()


class RNNClassifier(ClassifierLayer):
    def __init__(self):
        super(RNNClassifier, self).__init__()


class TransformerClassifier(ClassifierLayer):
    def __init__(self):
        super(TransformerClassifier, self).__init__()
