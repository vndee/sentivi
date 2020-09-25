import re
import string


class TextProcessor(object):
    def __init__(self):
        """
        A simple text processor base on regex
        """
        super(TextProcessor, self).__init__()
        self.__apply_function = list()

    def remove_punctuation(self):
        self.__apply_function.append(lambda x: re.sub(r'[' + string.punctuation + ']', '', x))

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
