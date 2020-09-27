import logging
import numpy as np

from typing import Optional
from sentivi.base_model import ClassifierLayer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report


class NaiveBayesClassifier(ClassifierLayer):
    def __init__(self, num_labels: int = 3):
        super(NaiveBayesClassifier, self).__init__()

        self.num_labels = num_labels
        self.clf = GaussianNB()

    def __call__(self, data, *args, **kwargs):
        (train_X, train_y), (test_X, test_y) = data
        assert train_X.shape[1:] == test_X.shape[1:], ValueError(
            f'Number of train features must be equal to test features: {train_X.shape[1:]} != {test_X.shape[1:]}')

        if train_X.shape.__len__() > 2:
            flatten_dim = 1
            for x in train_X.shape[1:]:
                flatten_dim *= x

            logging.info(f'Input features view be flatten into np.ndarray({train_X.shape[0]}, {flatten_dim}) '
                         f'for NaiveBayesClassifier.')
            train_X = train_X.reshape((train_X.shape[0], flatten_dim))
            test_X = test_X.reshape((test_X.shape[0], flatten_dim))

        # training
        self.clf.fit(train_X, train_y)
        predicts = self.clf.predict(train_X)
        train_report = classification_report(train_y, predicts)
        print(f'Training results:\n{train_report}')

        # testing
        predicts = self.clf.predict(test_X)
        test_report = classification_report(test_y, predicts)
        print(f'Test results:\n{test_report}')

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
