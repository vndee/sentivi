from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier


class DecisionTreeClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(DecisionTreeClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)
        self.clf = _DecisionTreeClassifier(**kwargs)
