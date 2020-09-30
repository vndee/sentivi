import torch
import numpy as np
from typing import Optional

from sentivi.base_model import ClassifierLayer
from torch.utils.data import Dataset, DataLoader
from sentivi.classifier.nn_clf import NeuralNetworkClassifier
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class TransformerClassifier(ClassifierLayer):
    class TransformerDataset(Dataset):
        def __init__(self, batch_encodings, labels):
            """
            Initialize transformer dataset

            :param batch_encodings:
            :param labels:
            """
            self.encodings = batch_encodings
            self.labels = labels

            assert self.encodings['input_ids'].__len__() == self.encodings['token_type_ids'].__len__(), ValueError(
                f'input_ids and token_type_ids must be in the same length.')
            assert self.encodings['input_ids'].__len__() == self.encodings['attention_mask'].__len__(), ValueError(
                f'input_ids and attention_mask must be in the same length.')
            assert self.encodings['input_ids'].__len__() == self.labels.__len__(), ValueError(
                f'input_ids and labels must be in the same length.')

        def __getitem__(self, item):
            return torch.from_numpy(np.array(self.encodings['input_ids'][item])), torch.from_numpy(
                np.array(self.encodings['token_type_ids'][item])), torch.from_numpy(
                np.array(self.encodings['attention_mask'][item])), torch.from_numpy(np.array(self.labels[item]))

        def __len__(self):
            return self.encodings['input_ids'].__len__()

    class TransformerPredictedDataset(Dataset):
        def __init__(self, batch_encodings):
            """
            Initialize transformer dataset

            :param batch_encodings:
            """
            self.encodings = batch_encodings

            assert self.encodings['input_ids'].__len__() == self.encodings['token_type_ids'].__len__(), ValueError(
                f'input_ids and token_type_ids must be in the same length.')
            assert self.encodings['input_ids'].__len__() == self.encodings['attention_mask'].__len__(), ValueError(
                f'input_ids and attention_mask must be in the same length.')

        def __getitem__(self, item):
            return torch.from_numpy(np.array(self.encodings['input_ids'][item])), torch.from_numpy(
                np.array(self.encodings['token_type_ids'][item])), torch.from_numpy(
                np.array(self.encodings['attention_mask'][item]))

        def __len__(self):
            return self.encodings['input_ids'].__len__()

    def __init__(self,
                 num_labels: Optional[int] = 3,
                 language_model_shortcut: Optional[str] = 'vinai/phobert',
                 freeze_language_model: Optional[bool] = True,
                 batch_size: Optional[int] = 2,
                 warmup_steps: Optional[int] = 100,
                 weight_decay: Optional[float] = 0.01,
                 accumulation_steps: Optional[int] = 50,
                 save_steps: Optional[int] = 100,
                 learning_rate: Optional[float] = 3e-5,
                 device: Optional[str] = 'cpu',
                 optimizer=None,
                 criterion=None,
                 num_epochs: Optional[int] = 10,
                 num_workers: Optional[int] = 2,
                 *args,
                 **kwargs):
        """
        Initialize TransformerClassifier instance

        :param num_labels: number of polarities
        :param language_model_shortcut: language model shortcut
        :param freeze_language_model: whether language model is freeze or not
        :param batch_size: training batch size
        :param warmup_steps: learning rate warm up step
        :param weight_decay: learning rate weight decay
        :param accumulation_steps: optimizer accumulation step
        :param save_steps: saving step
        :param learning_rate: training learning rate
        :param device: training and evaluating rate
        :param optimizer: training optimizer
        :param criterion: training criterion
        :param num_epochs: maximum number of epochs
        :param num_workers: number of DataLoader workers
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(TransformerClassifier, self).__init__()

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.language_model_shortcut = language_model_shortcut
        self.save_steps = save_steps
        self.device = device
        self.optimizer = optimizer
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.predict_loader = None

        self.clf_config = AutoConfig.from_pretrained(language_model_shortcut)
        self.clf_config.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(language_model_shortcut)
        self.clf = AutoModelForSequenceClassification.from_pretrained(language_model_shortcut, config=self.clf_config)

        if freeze_language_model is True:
            for param in self.clf.base_model.parameters():
                param.requires_grad = True

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
            for input_ids_X, token_type_ids_X, attention_mask_X, y in loader:
                input_ids_X = input_ids_X.to(self.device)
                attention_mask_X = attention_mask_X.to(self.device)
                y = y.to(self.device)
                loss, preds = self.clf(input_ids_X, attention_mask=attention_mask_X, labels=y)

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

    def forward(self, data, *args, **kwargs):
        """
        Training and evaluating TransformerClassifier instance

        :param data: TransformerTextEncoder output
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: training and evaluating results
        :rtype: str
        """
        self.no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.clf.named_parameters() if not any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.clf.named_parameters() if any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=kwargs.get('learning_rate', 1e-5))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=1000)

        (train_X, train_y), (test_X, test_y) = data

        self.learning_rate = kwargs.get('learning_rate', self.learning_rate)
        self.optimizer = kwargs.get('optimizer', self.optimizer)
        self.scheduler = kwargs.get('scheduler', self.scheduler)
        self.batch_size = kwargs.get('batch_size', self.batch_size)
        self.criterion = kwargs.get('criterion', torch.nn.CrossEntropyLoss())
        self.num_epochs = kwargs.get('num_epochs', self.num_epochs)

        self.train_loader = DataLoader(TransformerClassifier.TransformerDataset(train_X, train_y), shuffle=True,
                                       batch_size=self.batch_size, num_workers=self.num_workers)
        self.test_loader = DataLoader(TransformerClassifier.TransformerDataset(test_X, test_y), shuffle=True,
                                      batch_size=self.batch_size, num_workers=self.num_workers)

        self.clf = self.clf.to(self.device)
        len_train, len_test = train_X['input_ids'].__len__(), test_X['input_ids'].__len__()

        for epoch in range(self.num_epochs):
            self.clf.train()
            _preds, _targets = None, None
            train_loss, train_acc, train_f1, test_loss, test_acc, test_f1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            for idx, (input_ids_X, token_type_ids_X, attention_mask_X, y) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                input_ids_X = input_ids_X.to(self.device)
                attention_mask_X = attention_mask_X.to(self.device)
                y = y.to(self.device)
                loss, preds = self.clf(input_ids_X, attention_mask=attention_mask_X, labels=y)

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
                if idx % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()

                train_loss = train_loss + loss.item()

            train_acc, train_f1 = NeuralNetworkClassifier.compute_metrics(_preds, _targets)

            self.clf.eval()
            _preds, _targets = None, None
            with torch.no_grad():
                for input_ids_X, token_type_ids_X, attention_mask_X, y in self.test_loader:
                    input_ids_X = input_ids_X.to(self.device)
                    attention_mask_X = attention_mask_X.to(self.device)
                    y = y.to(self.device)
                    loss, preds = self.clf(input_ids_X, attention_mask=attention_mask_X, labels=y)

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

            print(f'[EPOCH {epoch + 1}/{self.num_epochs}] Train loss: {train_loss / len_train} | '
                  f'Train acc: {train_acc} | Train F1: {train_f1} | Test loss: {test_loss / len_test} | '
                  f'Test acc: {test_acc} | Test F1: {test_f1}')

        print('Finishing...')

        if 'save_path' in kwargs:
            self.save(kwargs['save_path'])

        return f'Train results:\n{self.get_overall_result(self.train_loader)}\n' \
               f'Test results:\n{self.get_overall_result(self.test_loader)}'

    def predict(self, X, *args, **kwargs):
        """
        Predict polarities with given list of sentences

        :param X: list of sentences
        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: list of polarities
        :rtype: str
        """

        _preds = None
        self.predict_loader = DataLoader(TransformerClassifier.TransformerPredictedDataset(X), shuffle=True,
                                         batch_size=self.batch_size, num_workers=self.num_workers)

        with torch.no_grad():
            for input_ids_X, token_type_ids_X, attention_mask_X in self.predict_loader:
                input_ids_X = input_ids_X.to(self.device)
                attention_mask_X = attention_mask_X.to(self.device)
                preds = self.clf(input_ids_X, attention_mask=attention_mask_X)[0]

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

    __call__ = forward
