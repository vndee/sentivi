from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(NaiveBayesClassifier, self).__init__(num_labels, *args, **kwargs)
        self.clf = GaussianNB(**kwargs)
