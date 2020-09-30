import torch
import numpy as np

from tqdm import tqdm
from typing import Optional

from torch.utils.data import Dataset, DataLoader
from sentivi.base_model import ClassifierLayer
from sklearn.metrics import classification_report


class NeuralNetworkClassifier(ClassifierLayer):
    """
    Neural Network Classifier
    """
    def __init__(self,
                 num_labels: int = 3,
                 embedding_size: Optional[int] = None,
                 max_length: Optional[int] = None,
                 device: Optional[str] = 'cpu',
                 num_epochs: Optional[int] = 10,
                 learning_rate: Optional[float] = 1e-3,
                 batch_size: Optional[int] = 2,
                 shuffle: Optional[bool] = True,
                 random_state: Optional[int] = 101,
                 hidden_size: Optional[int] = 512,
                 num_workers: Optional[int] = 2,
                 *args,
                 **kwargs):
        """
        Neural Network Classifier

        :param num_labels: number of polarities
        :param embedding_size: input embedding size
        :param max_length: maximum number of input text
        :param device: training device
        :param num_epochs: maximum number of epochs
        :param learning_rate: training learning rate
        :param batch_size: training batch size
        :param shuffle: whether DataLoader is shuffle or not
        :param random_state: random.seed
        :param hidden_size: hidden size
        :param num_workers: number of DataLoader workers
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(NeuralNetworkClassifier, self).__init__()

        self.num_labels = num_labels
        self.clf = None
        self.optimizer = None
        self.criterion = None
        self.learning_rate_scheduler = None
        self.device = None

        self.embedding_size = embedding_size
        self.max_length = max_length
        self.device = device
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.hidden_size = hidden_size
        self.num_workers = num_workers
        self.train_X = None
        self.train_y = None
        self.test_X = None
        self.test_y = None
        self.predict_loader = None
        self.train_loader = None
        self.test_loader = None

    def forward(self, data, *args, **kwargs):
        """
        Training and evaluating NeuralNetworkClassifier

        :param data: TextEncoder output
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: training and evaluating results
        :rtype: str
        """
        pass

    @staticmethod
    def compute_metrics(preds, targets, eval=False):
        """
        Compute accuracy and F1

        :param preds: prediction output
        :param targets: ground-truth value
        :param eval: whether is eval or not
        :return:
        """
        if eval is True:
            return classification_report(preds, targets, zero_division=1)

        report = classification_report(preds, targets, output_dict=True, zero_division=1)
        return report['accuracy'], report['macro avg']['f1-score']

    def get_overall_result(self, loader):
        """
        Get overall result

        :param loader: DataLoader
        :return: overall result
        :rtype: str
        """

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

        return NeuralNetworkClassifier.compute_metrics(_preds, _targets, eval=True)

    def fit(self, *args, **kwargs):
        """
        Feed-forward network

        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return:
        """
        assert isinstance(self.clf, torch.nn.Module), ValueError(
            f'Classifier model using Neural Network must be torch.nn.Module, not {self.clf}')

        self.clf = self.clf.to(self.device)

        self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
        self.optimizer = kwargs.get('optimizer', torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate))
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)
        self.num_workers = kwargs.get('num_workers', self.num_workers)

        self.train_loader = DataLoader(NeuralNetworkDataset(self.train_X, self.train_y), batch_size=self.batch_size,
                                       shuffle=self.shuffle, num_workers=self.num_workers)
        self.test_loader = DataLoader(NeuralNetworkDataset(self.test_X, self.test_y), batch_size=self.batch_size,
                                      shuffle=self.shuffle, num_workers=self.num_workers)

        for epoch in range(self.num_epochs):
            self.clf.train()
            _preds, _targets = None, None
            train_loss, train_acc, train_f1, test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for X, y in self.train_loader:
                self.optimizer.zero_grad()
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

            train_acc, train_f1 = NeuralNetworkClassifier.compute_metrics(_preds, _targets)

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

            test_acc, test_f1 = NeuralNetworkClassifier.compute_metrics(_preds, _targets)

            print(f'[EPOCH {epoch + 1}/{self.num_epochs}] Train loss: {train_loss / self.train_X.shape[0]} | '
                  f'Train acc: {train_acc} | Train F1: {train_f1} | Test loss: {test_loss / self.test_X.shape[0]} | '
                  f'Test acc: {test_acc} | Test F1: {test_f1}')

        print('Finishing...')

        if 'save_path' in kwargs:
            self.save(kwargs['save_path'])

        return f'Train results:\n{self.get_overall_result(self.train_loader)}\n' \
               f'Test results:\n{self.get_overall_result(self.test_loader)}'

    def _predict(self, X, *args, **kwargs):
        """
        Predict polarities with given list of sentences

        :param X: list of sentences
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: list of polarities
        :rtype: str
        """

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

        :param save_path: path to saved model
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return:
        """

        torch.save(self.clf.state_dict(), save_path)
        print(f'Saved classifier model to {save_path}')

    def load(self, model_path, *args, **kwargs):
        """
        Load model from disk

        :param model_path: path to model path
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return:
        """

        self.clf.load_state_dict(torch.load(model_path, map_location=self.device))
        print(f'Loaded classifier model to {model_path}')


class NeuralNetworkDataset(Dataset):
    def __init__(self, X, y, *args, **kwargs):
        super(NeuralNetworkDataset, self).__init__(*args, **kwargs)

        assert X.shape[0] == y.shape[0], ValueError(f'Number of samples must be equal {X.shape[0]} != {y.shape[0]}.')
        self.X, self.y = X, y

    def __getitem__(self, item):
        return torch.from_numpy(self.X[item]), torch.from_numpy(np.array(self.y[item]))

    def __len__(self):
        return self.X.shape[0]
