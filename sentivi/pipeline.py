from typing import Optional
from sentivi.data import DataLoader


class Pipeline(object):
    def __init__(self, *args, **kwargs):
        """
        Init full pipeline for Vietnamese Sentiment Analysis
        :param args:
        :param kwargs:
        """
        super(Pipeline, self).__init__(*args, **kwargs)
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

        self.__vocab = None
        self.__labels_set = None
        self.__n_grams = None

    def __call__(self, *args, **kwargs):
        """
        Execute all
        :param args:
        :param kwargs:
        :return:
        """
        x = None
        for method in self.apply_layers:
            x = method(x, *args, **kwargs)
            if isinstance(method, DataLoader):
                self.__n_grams, self.__vocab, self.__labels_set = method.n_grams, method.vocab, method.labels_set
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
            x = method.predict(x, vocab=self.__vocab, n_grams=self.__n_grams, *args, **kwargs)
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
