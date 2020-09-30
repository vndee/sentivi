import os
import logging

from typing import Optional
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier.nn_clf import NeuralNetworkClassifier
from sentivi.classifier.transformer import TransformerClassifier

try:
    import _pickle as pickle
except ModuleNotFoundError:
    import pickle


class Pipeline(object):
    """
    Pipeline instance
    """
    def __init__(self, *args, **kwargs):
        """
        Initialize Pipeline instance

        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        """
        super(Pipeline, self).__init__()
        self.apply_layers = list()
        language_model_shortcut = None

        for method in args:
            self.apply_layers.append(method)

            if isinstance(method, TransformerClassifier):
                language_model_shortcut = method.language_model_shortcut

        if language_model_shortcut is not None:
            for method in self.apply_layers:
                if isinstance(method, TextEncoder):
                    if method.encode_type != 'transformer':
                        logging.warning(f'Expected transformer encoder type for TextEncoder, '
                                        f'but got {method.encode_type}. It\'s will be implicit cast into transformer')
                    method.encode_type = 'transformer'
                    method.language_model_shortcut = language_model_shortcut
                    break

        self.__vocab = None
        self.__labels_set = None
        self.__n_grams = None
        self.__max_length = None
        self.__embedding_size = None

    def append(self, method):
        """
        Append a callable layer

        :param method: [DataLayer, ClassifierLayer]
        :return: None
        """
        self.apply_layers.append(method)

    def keyword_arguments(self):
        """
        Return pipeline's protected attribute and its value in form of dictionary.

        :return: key-value of protected attributes
        :rtype: Dictionary
        """
        return {attr[11:]: getattr(self, attr) for attr in dir(self) if
                attr[:10] == '_Pipeline_' and getattr(self, attr) is not None}

    def forward(self, *args, **kwargs):
        """
        Execute all callable layer in self.apply_layers

        :param args:
        :param kwargs:
        :return:
        """
        x = None
        for method in self.apply_layers:
            x = method(x, *args, **kwargs, **self.keyword_arguments())

            if isinstance(method, DataLoader):
                self.__n_grams, self.__vocab, self.__labels_set, self.__max_length = method.n_grams, method.vocab, \
                                                                                     method.labels_set, \
                                                                                     method.max_length

        return x

    def predict(self, x: Optional[list], *args, **kwargs):
        """
        Predict target polarity from list of given features

        :param x: List of input texts
        :param args: arbitrary positional arguments
        :param kwargs: arbitrary keyword arguments
        :return: List of labels corresponding to given input texts
        :rtype: List
        """
        for method in self.apply_layers:
            if isinstance(method, DataLoader):
                text_processor = method.text_processor
                x = [' '.join([_text for _text in text_processor(text).split(' ') if _text != '']) for text in x]
                continue
            x = method.predict(x, *args, **kwargs, **self.keyword_arguments())
        return x

    def decode_polarity(self, x: Optional[list]):
        """
        Decode numeric polarities into label polarities

        :param x: List of numeric polarities (i.e [0, 1, 2, 1, 0])
        :return: List of label polarities (i.e ['neg', 'neu', 'pos', 'neu', 'neg']
        :rtype: List
        """
        return [self.__labels_set[idx] for idx in x]

    def get_labels_set(self):
        """
        Get labels set

        :return: List of labels
        :rtype: List
        """
        return self.__labels_set

    def get_vocab(self):
        """
        Get vocabulary

        :return: Vocabulary in form of List
        :rtype: List
        """
        return self.__vocab

    def save(self, save_path: str):
        """
        Save model to disk

        :param save_path: path to saved model
        :return:
        """
        import dill
        with open(save_path, 'wb') as stream:
            dill.dump(self, stream)
            print(f'Saved model to {save_path}')

    @staticmethod
    def load(model_path: str):
        """
        Load model from disk

        :param model_path: path to pre-trained model
        :return:
        """
        import dill
        assert os.path.exists(model_path), FileNotFoundError(f'Could not found {model_path}')
        with open(model_path, 'rb') as stream:
            print(f'Loaded model from {model_path}')
            return dill.load(stream)

    def to(self, device):
        """
        To device

        :param device:
        :return:
        """
        for method in self.apply_layers:
            if isinstance(method, NeuralNetworkClassifier) or isinstance(method, TransformerClassifier):
                method.clf = method.clf.to(device)

    def get_server(self):
        """
        Serving model

        :return:
        """
        from sentivi.service import RESTServiceGateway

        return RESTServiceGateway(self, port=5000)

    __call__ = forward
