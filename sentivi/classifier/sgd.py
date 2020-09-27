from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.linear_model import SGDClassifier as _SGDClassifier


class SGDClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(SGDClassifier, self).__init__(num_labels, *args, **kwargs)
        self.clf = _SGDClassifier(**kwargs)
