from sklearn.neighbors import NearestCentroid as _NearestCentroid
from sentivi.classifier.sklearn_clf import ScikitLearnClassifier


class NearestCentroidClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        """
        Initialize NearestCentroidClassifier

        :param num_labels: number of polarities
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(NearestCentroidClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)
        self.clf = _NearestCentroid(**kwargs)
