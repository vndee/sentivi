import torch
import torch.nn as nn

from typing import Optional
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

        :param num_labels: number of polarities
        :param embedding_size: input embedding size
        :param max_length: maximum length of input text
        :param device: training device
        :param num_epochs: maximum number of epochs
        :param learning_rate: training learning rate
        :param batch_size: training batch size
        :param shuffle: whether DataLoader is shuffle or not
        :param random_state: random.seed
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(TextCNNClassifier, self).__init__(num_labels, embedding_size, max_length, device, num_epochs,
                                                learning_rate, batch_size, shuffle, random_state, *args, **kwargs)

    def forward(self, data, *args, **kwargs):
        """
        Training and evaluating method

        :param data: TextEncoder output
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: training and evaluating results
        :rtype: str
        """
        (train_X, train_y), (test_X, test_y) = data

        if 'embedding_size' in kwargs:
            self.embedding_size = kwargs['embedding_size']
        elif self.embedding_size is None:
            assert train_X[-1].shape == test_X[-1].shape, ValueError(
                'Feature embedding size of train set and test set must be equal.')
            self.embedding_size = train_X.shape[-1]

        self.max_length = kwargs['max_length']
        assert train_X.shape.__len__() == test_X.shape.__len__(), ValueError(
            'Number of dimension in train set and test set must be equal.')
        assert train_X.shape.__len__() <= 3, ValueError(
            'Expected array with number of dimension less or equal than 3.')
        if train_X.shape.__len__() == 3:
            self.max_length = train_X.shape[1]
            self.train_X = train_X.reshape((train_X.shape[-3], 1, train_X.shape[-2], train_X.shape[-1]))
            self.test_X = test_X.reshape((test_X.shape[-3], 1, test_X.shape[-2], test_X.shape[-1]))
            print(f'Reshape input array into (n_samples, 1, 1, feature_dim) for TextCNN Network Classifier')
        else:
            self.max_length = 1
            self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[-1]))
            self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[-1]))
            print(f'Reshape input array into (n_samples, 1, feature_dim) for TextCNN Network Classifier')

        self.train_y, self.test_y = train_y, test_y

        if 'device' in kwargs:
            self.device = kwargs['device']

        self.clf = TextCNN(num_labels=self.num_labels, embedding_size=self.embedding_size, max_length=self.max_length)
        return self.fit(*args, **kwargs)

    def predict(self, X, *args, **kwargs):
        """
        Predict polarity with given sentences

        :param X: TextEncoder.predict output
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: list of numeric polarities
        :rtype: list
        """
        self.clf.eval()
        if X.shape.__len__() == 3:
            X = X.reshape((X.shape[-3], 1, X.shape[-2], X.shape[-1]))
        else:
            X = X.reshape((X.shape[0], 1, X.shape[-1]))

        return self._predict(X)

    __call__ = forward
