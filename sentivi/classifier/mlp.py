from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sentivi.classifier.sklearn_clf import ScikitLearnClassifier


class MLPClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        """
        Initialize MLPClassifier

        :param num_labels: number of polarities
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(MLPClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)
        self.clf = _MLPClassifier(**kwargs)
