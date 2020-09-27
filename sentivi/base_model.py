import numpy as np


class SentimentModel(object):
    def __init__(self):
        super(SentimentModel, self).__init__()

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class DataLayer(object):
    def __init__(self):
        super(DataLayer, self).__init__()

    def __call__(self, *args, **kwargs):
        pass


class ClassifierLayer(object):
    def __init__(self):
        super(ClassifierLayer, self).__init__()

    def __call__(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass

    def load(self, *args, **kwargs):
        pass


class PretrainedClassifier(object):
    def __init__(self, *args, **kwargs):
        super(PretrainedClassifier, self).__init__()
        self.classifier_type = None
        self.clf = None

    def save(self, save_path, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass
