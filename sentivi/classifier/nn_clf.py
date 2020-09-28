import torch

from sentivi.base_model import ClassifierLayer
from sklearn.metrics import classification_report


class NeuralNetworkClassifier(ClassifierLayer):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(NeuralNetworkClassifier, self).__init__()

        self.num_labels = num_labels
        self.clf = None
        self.optimizer = None
        self.criterion = None
        self.learning_rate_scheduler = None

    def __call__(self, data, *args, **kwargs):
        pass

    def predict(self, x, *args, **kwargs):
        pass

    def save(self, save_path, *args, **kwargs):
        pass

    def load(self, model_path, *args, **kwargs):
        pass
