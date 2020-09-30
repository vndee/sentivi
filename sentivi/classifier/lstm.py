import torch
import torch.nn as nn

from typing import Optional
from sentivi.classifier.nn_clf import NeuralNetworkClassifier


class LSTM(nn.Module):
    def __init__(self,
                 num_labels: int,
                 embedding_size: int,
                 hidden_size: int,
                 bidirectional: bool = False,
                 attention: bool = False,
                 hidden_layers: int = 1):
        """
        Initialize LSTM instance

        :param num_labels:
        :param embedding_size:
        :param hidden_size:
        :param bidirectional:
        :param attention:
        :param hidden_layers:
        """
        super(LSTM, self).__init__()

        self.num_labels = num_labels
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention = attention
        self.bidirectional = bidirectional
        self.hidden_layers = hidden_layers

        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_size, bidirectional=self.bidirectional,
                                  batch_first=True, num_layers=self.hidden_layers)
        self.linear = nn.Linear(self.hidden_size * (2 if self.bidirectional is True else 1), self.num_labels)

    def attention_layer(self, lstm_output, final_state):
        """
        Attention Layer
        :param lstm_output:
        :param final_state:
        :return:
        """
        hidden = final_state.view(-1, self.hidden_size * (2 if self.bidirectional is True else 1), 1)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        soft_attn_weights = torch.nn.functional.softmax(attn_weights, 1)
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, inputs):
        """
        Forward method for torch.nn.Module
        :param inputs:
        :return:
        """
        hidden_state = torch.autograd.Variable(
            torch.zeros(self.hidden_layers * (2 if self.bidirectional is True else 1), inputs.shape[0],
                        self.hidden_size, device=inputs.device))
        cell_state = torch.autograd.Variable(
            torch.zeros(self.hidden_layers * (2 if self.bidirectional is True else 1), inputs.shape[0],
                        self.hidden_size, device=inputs.device))

        output, (final_hidden_state, final_cell_state) = self.lstm(inputs, (hidden_state, cell_state))

        if self.attention is True:
            return self.linear(
                self.attention_layer(output, final_hidden_state[-2 if self.bidirectional is True else -1:]))
        else:
            final_hidden_state = final_hidden_state[-2 if self.bidirectional is True else -1:].permute(1, 0, 2)
            return self.linear(final_hidden_state.reshape(
                (final_hidden_state.shape[0], final_hidden_state.shape[1] * final_hidden_state.shape[2])))


class LSTMClassifier(NeuralNetworkClassifier):
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
                 hidden_size: Optional[int] = 512,
                 hidden_layers: Optional[int] = 2,
                 bidirectional: Optional[bool] = False,
                 attention: Optional[bool] = True,
                 *args,
                 **kwargs):
        """
        Initialize LSTMClassifier

        :param num_labels: number of polarities
        :param embedding_size: input embeddings' size
        :param max_length: maximum length of input text
        :param device: training device
        :param num_epochs: maximum number of epochs
        :param learning_rate: model learning rate
        :param batch_size: training batch size
        :param shuffle: whether DataLoader is shuffle or not
        :param random_state: random.seed number
        :param hidden_size: Long Short Term Memory hidden size
        :param bidirectional: whether to use BiLSTM or not
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(LSTMClassifier, self).__init__(num_labels, embedding_size, max_length, device, num_epochs, learning_rate,
                                             batch_size, shuffle, random_state, hidden_size, hidden_layers, attention,
                                             *args, **kwargs)

        self.bidirectional = bidirectional
        self.attention = attention
        self.hidden_layers = hidden_layers

    def forward(self, data, *args, **kwargs):
        """
        Training and evaluating methods

        :param data: TextEncoder output
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: training results
        """
        (train_X, train_y), (test_X, test_y) = data

        if 'embedding_size' in kwargs:
            self.embedding_size = kwargs['embedding_size']
        elif self.embedding_size is None:
            assert train_X[-1].shape == test_X[-1].shape, ValueError(
                'Feature embedding size of train set and test set must be equal.')
            self.embedding_size = train_X.shape[-1]

        assert train_X.shape.__len__() == test_X.shape.__len__(), ValueError(
            'Number of dimension in train set and test set must be equal.')
        assert train_X.shape.__len__() <= 3, ValueError(
            'Expected array with number of dimension less or equal than 3.')
        if train_X.shape.__len__() == 3:
            self.max_length = train_X.shape[1]
            self.train_X, self.test_X = train_X, test_X
        else:
            self.max_length = 1
            self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[-1]))
            self.test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[-1]))
            print(f'Reshape input array into (n_samples, 1, feature_dim) for LSTM Network Classifier')

        self.train_y, self.test_y = train_y, test_y

        if 'device' in kwargs:
            self.device = kwargs['device']

        self.clf = LSTM(num_labels=self.num_labels, embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                        bidirectional=self.bidirectional, attention=self.attention, hidden_layers=self.hidden_layers)
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
        if X.shape.__len__() == 2:
            X = X.reshape((X.shape[0], 1, X.shape[-1]))

        return self._predict(X)

    __call__ = forward
