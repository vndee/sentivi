import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Optional
from sklearn.metrics import classification_report

from sentivi.classifier.nn_clf import NeuralNetworkDataset
from sentivi.classifier.nn_clf import NeuralNetworkClassifier


class TextCNN(nn.Module):
    def __init__(self, num_labels: int, embedding_size: int, max_length: int):
        """
        Initialize TextCNN classifier
        :param num_labels:
        :param embedding_size:
        :param max_length:
        """
        super(TextCNN, self).__init__()

        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.max_length = max_length

        if self.max_length == 1:
            self.conv_1 = nn.Conv1d(1, 1, 3)
            self.conv_2 = nn.Conv1d(1, 1, 4)
            self.conv_3 = nn.Conv1d(1, 1, 5)

            self.max_pool_1 = nn.MaxPool1d(kernel_size=self.embedding_size - 3 + 1, stride=1)
            self.max_pool_2 = nn.MaxPool1d(kernel_size=self.embedding_size - 4 + 1, stride=1)
            self.max_pool_3 = nn.MaxPool1d(kernel_size=self.embedding_size - 5 + 1, stride=1)
        else:
            self.conv_1 = nn.Conv2d(1, 1, (3, self.embedding_size))
            self.conv_2 = nn.Conv2d(1, 1, (4, self.embedding_size))
            self.conv_3 = nn.Conv2d(1, 1, (5, self.embedding_size))

            self.max_pool_1 = nn.MaxPool2d((self.max_length - 3 + 1, 1))
            self.max_pool_2 = nn.MaxPool2d((self.max_length - 4 + 1, 1))
            self.max_pool_3 = nn.MaxPool2d((self.max_length - 5 + 1, 1))

        self.linear = nn.Linear(3, self.num_labels)

    def forward(self, x):
        """
        Forward method for torch.nn.Module
        :param x:
        :return:
        """
        batch = x.shape[0]

        x_1 = torch.nn.functional.relu(self.conv_1(x))
        x_2 = torch.nn.functional.relu(self.conv_2(x))
        x_3 = torch.nn.functional.relu(self.conv_3(x))

        x_1 = self.max_pool_1(x_1)
        x_2 = self.max_pool_2(x_2)
        x_3 = self.max_pool_3(x_3)

        x = torch.cat((x_1, x_2, x_3), -1)
        x = x.view(batch, 1, -1)

        x = self.linear(x)
        x = x.view(-1, self.num_labels)

        return x


class TextCNNClassifier(NeuralNetworkClassifier):
    def __init__(self,
                 num_labels: int,
                 embedding_size: Optional[int] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = 'cpu',
                 num_epochs: Optional[int] = 10,
                 learning_rate: Optional[float] = 1e-3,
                 batch_size: Optional[int] = 2,
                 shuffle: Optional[bool] = True,
                 random_state: Optional[int] = 101,
                 *args,
                 **kwargs):
        """
        Initialize TextCNNClassifier
        :param num_labels:
        :param embedding_size:
        :param max_length:
        :param device:
        :param num_epochs:
        :param learning_rate:
        :param batch_size:
        :param shuffle
        :param random_state
        :param args:
        :param kwargs:
        """
        super(TextCNNClassifier, self).__init__(num_labels, *args, **kwargs)

        self.embedding_size = embedding_size
        self.max_length = max_length
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.predict_loader = None
        self.train_loader = None
        self.test_loader = None

    @staticmethod
    def compute_metrics(preds, targets, eval=False):
        if eval is True:
            return classification_report(preds, targets, zero_division=1)

        report = classification_report(preds, targets, output_dict=True, zero_division=1)
        return report['accuracy'], report['macro avg']['f1-score']

    def get_overall_result(self, loader):
        self.clf.eval()
        _preds, _targets = None, None

        with torch.no_grad():
            for X, y in loader:
                X, y = X.type(torch.FloatTensor).to(self.device), y.type(torch.LongTensor).to(self.device)
                preds = self.clf(X)

                if self.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    y = y.detach().numpy()

                predicted = np.argmax(preds, -1)
                _preds = np.atleast_1d(predicted) if _preds is None else np.concatenate(
                    [_preds, np.atleast_1d(predicted)])
                _targets = np.atleast_1d(y) if _targets is None else np.concatenate([_targets, np.atleast_1d(y)])

        return TextCNNClassifier.compute_metrics(_preds, _targets, eval=True)

    def __call__(self, data, *args, **kwargs):
        """
        Training method
        :param data:
        :param args:
        :param kwargs:
        :return:
        """
        (train_X, train_y), (test_X, test_y) = data

        if 'embedding_size' in kwargs:
            self.embedding_size = kwargs['embedding_size']
        elif self.embedding_size is None:
            assert train_X[-1].shape == test_X[-1].shape, ValueError(
                'Feature embedding size of train set and test set must be equal.')
            self.embedding_size = train_X.shape[-1]

        if 'max_length' in kwargs:
            self.max_length = kwargs['max_length']
        elif self.max_length is None:
            assert train_X.shape.__len__() == test_X.shape.__len__(), ValueError(
                'Number of dimension in train set and test set must be equal.')
            assert train_X.shape.__len__() <= 3, ValueError(
                'Expected array with number of dimension less or equal than 3.')
            if train_X.shape.__len__() == 3:
                self.max_length = train_X.shape[1]
                train_X = train_X.reshape((train_X.shape[-3], 1, train_X.shape[-2], train_X.shape[-1]))
                test_X = test_X.reshape((test_X.shape[-3], 1, test_X.shape[-2], test_X.shape[-1]))
            else:
                self.max_length = 1
                train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[-1]))
                test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[-1]))
            print(f'Reshape input array into (n_samples, 1, 1, feature_dim) for Neural Network Classifier')

        if 'device' in kwargs:
            self.device = kwargs['device']

        self.clf = TextCNN(num_labels=self.num_labels, embedding_size=self.embedding_size, max_length=self.max_length)
        self.clf = self.clf.to(self.device)

        self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate))
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)

        self.train_loader = DataLoader(NeuralNetworkDataset(train_X, train_y), batch_size=self.batch_size,
                                       shuffle=self.shuffle)
        self.test_loader = DataLoader(NeuralNetworkDataset(test_X, test_y), batch_size=self.batch_size,
                                      shuffle=self.shuffle)

        for epoch in range(self.num_epochs):
            self.clf.train()
            _preds, _targets = None, None
            train_loss, train_acc, train_f1, test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for X, y in self.train_loader:
                X, y = X.type(torch.FloatTensor).to(self.device), y.type(torch.LongTensor).to(self.device)
                preds = self.clf(X)
                loss = self.criterion(preds, y)

                if self.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                    y = y.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()
                    y = y.detach().numpy()

                predicted = np.argmax(preds, -1)
                _preds = np.atleast_1d(predicted) if _preds is None else np.concatenate(
                    [_preds, np.atleast_1d(predicted)])
                _targets = np.atleast_1d(y) if _targets is None else np.concatenate([_targets, np.atleast_1d(y)])

                loss.backward()
                self.optimizer.step()

                train_loss = train_loss + loss.item()

            train_acc, train_f1 = TextCNNClassifier.compute_metrics(_preds, _targets)

            self.clf.eval()
            _preds, _targets = None, None
            with torch.no_grad():
                for X, y in self.test_loader:
                    X, y = X.type(torch.FloatTensor).to(self.device), y.type(torch.LongTensor).to(self.device)
                    preds = self.clf(X)
                    loss = self.criterion(preds, y)

                    if self.device == 'cuda':
                        preds = preds.detach().cpu().numpy()
                        y = y.detach().cpu().numpy()
                    else:
                        preds = preds.detach().numpy()
                        y = y.detach().numpy()

                    predicted = np.argmax(preds, -1)
                    _preds = np.atleast_1d(predicted) if _preds is None else np.concatenate(
                        [_preds, np.atleast_1d(predicted)])
                    _targets = np.atleast_1d(y) if _targets is None else np.concatenate([_targets, np.atleast_1d(y)])

                    test_loss = test_loss + loss.item()

            test_acc, test_f1 = TextCNNClassifier.compute_metrics(_preds, _targets)

            print(f'[EPOCH {epoch + 1}/{self.num_epochs}] Train loss: {train_loss / train_X.shape[0]} | '
                  f'Train acc: {train_acc} | Train F1: {train_f1} | Test loss: {test_loss / test_X.shape[0]} | '
                  f'Test acc: {test_acc} | Test F1: {test_f1}')

        print('Finishing...')

        return f'Train results:\n{self.get_overall_result(self.train_loader)}\n' \
               f'Test results:\n{self.get_overall_result(self.test_loader)}'

    def predict(self, X, *args, **kwargs):
        """
        Predict polarity with given sentences
        :param X:
        :param args:
        :param kwargs:
        :return:
        """
        self.clf.eval()
        if X.shape.__len__() == 3:
            X = X.reshape((X.shape[-3], 1, X.shape[-2], X.shape[-1]))
        else:
            X = X.reshape((X.shape[0], 1, X.shape[-1]))

        _preds = None
        self.predict_loader = DataLoader(X, batch_size=self.batch_size, shuffle=self.shuffle)

        with torch.no_grad():
            for items in tqdm(self.predict_loader, desc='Prediction'):
                items = items.type(torch.FloatTensor).to(self.device)
                preds = self.clf(items)

                if self.device == 'cuda':
                    preds = preds.detach().cpu().numpy()
                else:
                    preds = preds.detach().numpy()

                predicted = np.argmax(preds, -1)
                _preds = np.atleast_1d(predicted) if _preds is None else np.concatenate(
                    [_preds, np.atleast_1d(predicted)])

        return _preds

    def save(self, save_path, *args, **kwargs):
        """
        Save model to disk
        :param save_path:
        :param args:
        :param kwargs:
        :return:
        """
        torch.save(self.clf.state_dict(), save_path)

    def load(self, model_path, *args, **kwargs):
        """
        Load model from disk
        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        self.clf.load_state_dict(torch.load(model_path, map_location=self.device))
