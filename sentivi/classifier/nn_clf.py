import torch
import numpy as np

from torch.utils.data import Dataset
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
        self.device = None

    def __call__(self, data, *args, **kwargs):
        pass

    def predict(self, x, *args, **kwargs):
        pass

    def save(self, save_path, *args, **kwargs):
        pass

    def load(self, model_path, *args, **kwargs):
        pass


class NeuralNetworkDataset(Dataset):
    def __init__(self, X, y, *args, **kwargs):
        super(NeuralNetworkDataset, self).__init__()

        assert X.shape[0] == y.shape[0], ValueError('Number of samples must be equal.')
        self.X, self.y = X, y

    def __getitem__(self, item):
        return torch.from_numpy(self.X[item]), torch.from_numpy(np.array(self.y[item]))

    def __len__(self):
        return self.X.shape[0]
