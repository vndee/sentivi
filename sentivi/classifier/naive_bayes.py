from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.naive_bayes import GaussianNB


class NaiveBayesClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(NaiveBayesClassifier, self).__init__(num_labels, *args, **kwargs)
        priors = kwargs.get('priors', None)
        var_smoothing = kwargs.get('var_smoothing', 1e-9)
        self.clf = GaussianNB(priors=priors, var_smoothing=var_smoothing)
