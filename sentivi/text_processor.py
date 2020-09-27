import re
import logging

from typing import Optional
from pyvi import ViTokenizer


class TextProcessor(object):
    def __init__(self, methods: Optional[list] = None):
        """
        A simple text processor base on regex
        """
        super(TextProcessor, self).__init__()
        self.__apply_function = list()

        if methods is not None:
            for method in methods:
                if hasattr(self, method):
                    getattr(self, method)()
                else:
                    logging.warning(f'There is no text processor method: {method}. Therefore {method} will be ignored.')

    def lower(self):
        self.__apply_function.append(lambda x: x.lower())

    def capitalize_first(self):
        self.__apply_function.append(lambda x: x.title())

    def capitalize(self):
        self.__apply_function.append(lambda x: x.upper())

    def word_segmentation(self):
        self.__apply_function.append(lambda x: ViTokenizer.tokenize(x))

    def remove_punctuation(self):
        self.__apply_function.append(lambda x: re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~]', '', x))

    def add_pattern(self, pattern, replace_text):
        self.__apply_function.append(lambda x: re.sub(pattern, replace_text, x))

    def __call__(self, _text):
        for function in self.__apply_function:
            _text = function(_text)
        return _text

    @staticmethod
    def n_gram_split(_x, _n_grams):
        _x = _x.split(' ')
        words = list()
        for idx, _ in enumerate(_x):
            if idx + _n_grams > _x.__len__():
                break
            words.append(' '.join(_x[idx: idx + _n_grams]))

        return words
