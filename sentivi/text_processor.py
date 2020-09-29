import re
import logging

from typing import Optional
from pyvi import ViTokenizer


class TextProcessor(object):
    """
        A simple text processor base on regex
    """

    def __init__(self, methods: Optional[list] = None):
        """
        Initialize TextProcessor instance

        :param methods: list of text preprocessor methods need to be applied, for example: ['remove_punctuation',
                        'word_segmentation']
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
        """
        Lower text

        :return:
        """
        self.__apply_function.append(lambda x: x.lower())

    def capitalize_first(self):
        """
        Capitalize first letter of a given text

        :return:
        """
        self.__apply_function.append(lambda x: x.title())

    def capitalize(self):
        """
        It is equivalent to str.upper()

        :return:
        """

        self.__apply_function.append(lambda x: x.upper())

    def word_segmentation(self):
        """
        Using PyVi to tokenize Vietnamese text. Note that this feature only use for Vietnamese text analysis.

        :return:
        """

        self.__apply_function.append(lambda x: ViTokenizer.tokenize(x))

    def remove_punctuation(self):
        """
        Remove punctuation of given text

        :return:
        """

        self.__apply_function.append(lambda x: re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\]^`{|}~]', '', x))

    def add_pattern(self, pattern, replace_text):
        """
        It is equivalent to re.sub()

        :param pattern: regex pattern
        :param replace_text: replace text
        :return:
        """

        self.__apply_function.append(lambda x: re.sub(pattern, replace_text, x))

    def add_method(self, method):
        """
        Add your method into TextProcessor

        :param method: Lambda function
        :return:
        """
        self.__apply_function.append(method)

    def __call__(self, _text):
        for function in self.__apply_function:
            _text = function(_text)
        return re.sub(' +', ' ', _text)

    @staticmethod
    def n_gram_split(_x, _n_grams):
        """
        Split text into n-grams form

        :param _x: Input text
        :param _n_grams: n-grams
        :return: List of words
        :rtype: List
        """

        _x = _x.split(' ')
        words = list()
        for idx, _ in enumerate(_x):
            if idx + _n_grams > _x.__len__():
                break
            words.append(' '.join(_x[idx: idx + _n_grams]))

        return words
