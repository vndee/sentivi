from sklearn.svm import SVC
from sentivi.classifier.sklearn_clf import ScikitLearnClassifier


class SVMClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(SVMClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)
        self.clf = SVC(**kwargs)
