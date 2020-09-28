from typing import Optional
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier.transformer import TransformerClassifier


class Pipeline(object):
    def __init__(self, *args):
        """
        Init full pipeline for Vietnamese Sentiment Analysis
        :param args:
        :param kwargs:
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
                    method.encode_type = 'transformer'
                    method.language_model_shortcut = language_model_shortcut
                    break

        self.__vocab = None
        self.__labels_set = None
        self.__n_grams = None
        self.__max_length = None
        self.__embedding_size = None

    def keyword_arguments(self):
        return {attr[11:]: getattr(self, attr) for attr in dir(self) if
                attr[:10] == '_Pipeline_' and getattr(self, attr) is not None}

    def __call__(self, *args, **kwargs):
        """
        Execute all
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
        :param x:
        :param args:
        :param kwargs:
        :return:
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
        Decode numeric targets into label targets
        :param x:
        :return:
        """
        return [self.__labels_set[idx] for idx in x]

    def get_labels_set(self):
        """
        Get labels set
        :return:
        """
        return self.__labels_set

    def get_vocab(self):
        """
        Get vocabulary
        :return:
        """
        return self.__vocab
