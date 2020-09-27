from typing import Optional

from sentivi import PretrainedClassifier
from sklearn.metrics import classification_report


class Pipeline(object):
    def __init__(self, *args):
        super(Pipeline, self).__init__()
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

    def __call__(self, *args, **kwargs):
        x: Optional[PretrainedClassifier] = None
        for method in self.apply_layers:
            x = method(x, *args, **kwargs)
        return x

    def test(self, model: PretrainedClassifier, x):
        for method in self.apply_layers:
            x = method(x)
