import logging
import numpy as np

from typing import Optional
from sentivi import ClassifierLayer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class NaiveBayesClassifier(ClassifierLayer):
    def __init__(self, num_labels: int = 3):
        super(NaiveBayesClassifier, self).__init__()

        self.num_labels = num_labels
        self.clf = GaussianNB()

    def __call__(self, data: (np.ndarray, np.ndarray), *args, **kwargs):
        X, y = data
        if X.shape.__len__() > 2:
            flatten_dim = 1
            for x in X.shape[1:]:
                flatten_dim *= x

            logging.info(
                f'Input features view be flatten into np.ndarray({X.shape[0]}, {flatten_dim}) for NaiveBayesClassifier.')
            X = X.reshape((X.shape[0], flatten_dim))

        if kwargs['mode'] == 'train':
            self.clf.fit(X, y)
            predicts = self.clf.predict(X)
            report = classification_report(y, predicts)
            print(f'Training results:\n{report}')
            if 'save_path' in kwargs:
                self.save(kwargs['save_path'])

    def save(self, save_path, *args, **kwargs):
        """
        Dump model to disk
        :param save_path:
        :param args:
        :param kwargs:
        :return:
        """
        import pickle
        with open(save_path, 'wb') as file:
            pickle.dump(self.clf, file)
            print(f'Saved classifier model to {save_path}')

    def load(self, model_path, *args, **kwargs):
        """
        Load model from disk
        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        import pickle
        with open(model_path, 'rb') as file:
            self.clf = pickle.load(model_path)
            print(f'Loaded pretrained model from {model_path}.')
