import math
import logging
import numpy as np

from typing import Optional
from sentivi.data import DataLayer
from sentivi.data.data_loader import Corpus
from sentivi.text_processor import TextProcessor

from sklearn.feature_extraction.text import TfidfVectorizer


class TextEncoder(DataLayer):
    def __init__(self, encode_type: Optional[str] = None, max_length: Optional[int] = None):
        """
        Simple text encode layer
        :param encode_type: ['one_hot', 'word2vec', 'bow', 'tf-idf']
        """
        super(TextEncoder, self).__init__()

        if encode_type is None:
            logging.warning(f'encode_type will be set to default value one_hot.')

        if max_length is None:
            logging.warning(f'max_length will be set to default value 256')

        self.encode_type = encode_type
        self.max_length = 256

    def __call__(self, x: Corpus) -> (np.ndarray, np.ndarray):
        target = np.array([y for x, y in x])
        if self.encode_type == 'one_hot':
            return self.one_hot(x), target
        elif self.encode_type == 'bow':
            return self.bow(x), target
        elif self.encode_type == 'tf_idf':
            return self.tf_idf(x), target

    def one_hot(self, x: Corpus) -> np.ndarray:
        """
        Convert corpus into batch of one-hot vectors.
        :param x:
        :return:
        """
        vocab = x.vocab
        _x = np.zeros((x.__len__(), self.max_length, vocab.__len__()))
        for i, (item, _) in enumerate(x):
            items = TextProcessor.n_gram_split(item, x.n_grams)
            for j, token in enumerate(items):
                if j >= self.max_length:
                    break
                idx = vocab.index(token)
                _x[i][j][idx] = 1

        return _x

    def bow(self, x: Corpus) -> np.ndarray:
        """
        Bag-of-Word encoder
        :return:
        """
        vocab = x.vocab
        _x = np.zeros((x.__len__(), vocab.__len__()))
        for i, (item, _) in enumerate(x):
            items = TextProcessor.n_gram_split(item, x.n_grams)
            for token in items:
                j = vocab.index(token)
                _x[i][j] = _x[i][j] + 1

        return _x

    def tf_idf(self, x: Corpus) -> np.ndarray:
        """
        Simple TF-IDF feature
        :param x:
        :return:
        """
        vectorizer = TfidfVectorizer()
        items = [TextProcessor.n_gram_split(item, x.n_grams) for item, _ in x]
        appearances_in_doc = {k: 0 for k in x.vocab}

        for _ in items:
            _set = set()
            for __ in _:
                if __ not in _set:
                    appearances_in_doc[__] += 1
                    _set.add(__)

        vocab = x.vocab
        _x = np.zeros((x.__len__(), vocab.__len__()))
        for i, _ in enumerate(items):
            appearances_in_here = dict()
            for __ in _:
                if __ not in appearances_in_here:
                    appearances_in_here[__] = 1
                else:
                    appearances_in_here[__] += 1

            for __ in _:
                j = vocab.index(__)
                _x[i][j] = math.log(1 + appearances_in_here[__]) * math.log(x.__len__() / appearances_in_doc[__])

        return _x
