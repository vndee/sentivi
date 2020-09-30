from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier as _GaussianProcessClassifier


class GaussianProcessClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        """
        Initialize GaussianProcessClassifier

        :param num_labels: number of polarities
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(GaussianProcessClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)

        if 'kernel' not in kwargs:
            kwargs['kernel'] = 1.0 * RBF(1.0)

        self.clf = _GaussianProcessClassifier(**kwargs)
