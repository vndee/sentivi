from sentivi import ClassifierLayer
from .naive_bayes import NaiveBayesClassifier


class SVMClassifier(ClassifierLayer):
    def __init__(self):
        super(SVMClassifier, self).__init__()


class CNNClassifier(ClassifierLayer):
    def __init__(self):
        super(CNNClassifier, self).__init__()


class RNNClassifier(ClassifierLayer):
    def __init__(self):
        super(RNNClassifier, self).__init__()


class TransformerClassifier(ClassifierLayer):
    def __init__(self):
        super(TransformerClassifier, self).__init__()


class MLPClassifier(ClassifierLayer):
    def __init__(self):
        super(MLPClassifier, self).__init__()


class DecisionTreeClassifier(ClassifierLayer):
    def __init__(self):
        super(DecisionTreeClassifier, self).__init__()
