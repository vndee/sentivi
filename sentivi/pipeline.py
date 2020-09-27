from typing import Optional
from sentivi.data import DataLoader


class Pipeline(object):
    def __init__(self, *args):
        super(Pipeline, self).__init__()
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

        self.__vocab = None
        self.__labels_set = None
        self.__n_grams = None

    def __call__(self, *args, **kwargs):
        x = None
        for method in self.apply_layers:
            x = method(x, *args, **kwargs)
            if isinstance(method, DataLoader):
                self.__n_grams, self.__vocab, self.__labels_set = method.n_grams, method.vocab, method.labels_set
        return x

    def predict(self, x: Optional[list], *args, **kwargs):
        for method in self.apply_layers:
            if isinstance(method, DataLoader):
                text_processor = method.text_processor
                x = [' '.join([_text for _text in text_processor(text).split(' ') if _text != '']) for text in x]
                continue
            x = method.predict(x, vocab=self.__vocab, n_grams=self.__n_grams, *args, **kwargs)
        return x

    def decode_polarity(self, x: Optional[list]):
        return [self.__labels_set[idx] for idx in x]

    def get_labels_set(self):
        return self.__labels_set

    def get_vocab(self):
        return self.__vocab
