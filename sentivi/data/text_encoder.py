import logging
import numpy as np

from tqdm import tqdm
from typing import Optional
from sentivi.data import DataLayer
from sentivi.data.data_loader import Corpus
from sentivi.text_processor import TextProcessor


class TextEncoder(DataLayer):
    def __init__(self,
                 encode_type: Optional[str] = None,
                 max_length: Optional[int] = None,
                 model_path: Optional[str] = None):
        """
        Simple text encode layer
        :param encode_type: ['one-hot', 'word2vec', 'bow', 'tf-idf']
        """
        super(TextEncoder, self).__init__()

        if encode_type is None:
            logging.warning(f'encode_type will be set to default value one-hot.')

        if max_length is None:
            logging.warning(f'max_length will be set to default value 256')

        self.encode_type = encode_type
        self.max_length = 256
        self.model_path = model_path
        self.word_vectors = None

        assert self.encode_type in ['one-hot', 'word2vec', 'bow', 'tf-idf'], ValueError(
            'Text encoder type must be one of [\'one-hot\', \'word2vec\', \'bow\', \'tf-idf\']')

    def __call__(self, x, *args, **kwargs) -> (np.ndarray, np.ndarray):
        if 'mode' in kwargs:
            if kwargs['mode'] == 'predict':
                assert 'n_grams' in kwargs, AttributeError('It\'s seem like DataLoader has no n_grams attribute.')
                assert 'vocab' in kwargs, AttributeError('It\'s seem like DataLoader has no vocab attribute.')

                if self.encode_type == 'one-hot':
                    return self.one_hot(x, kwargs['vocab'], kwargs['n_grams'])
                elif self.encode_type == 'bow':
                    return self.bow(x, kwargs['vocab'], kwargs['n_grams'])
                elif self.encode_type == 'tf-idf':
                    return self.tf_idf(x, kwargs['vocab'], kwargs['n_grams'])
                elif self.encode_type == 'word2vec':
                    return self.word2vec(x, kwargs['vocab'], kwargs['n_grams'])

        train_set, test_set = x.get_train_set(), x.get_test_set()
        train_X, train_y = train_set
        test_X, test_y = test_set

        train_y, test_y = np.array(train_y), np.array(test_y)
        vocab, n_grams = x.vocab, x.n_grams

        if self.encode_type == 'one-hot':
            return (self.one_hot(train_X, vocab, n_grams), train_y), (self.one_hot(test_X, vocab, n_grams), test_y)
        elif self.encode_type == 'bow':
            return (self.bow(train_X, vocab, n_grams), train_y), (self.bow(test_X, vocab, n_grams), test_y)
        elif self.encode_type == 'tf-idf':
            return (self.tf_idf(train_X, vocab, n_grams), train_y), (self.tf_idf(test_X, vocab, n_grams), test_y)
        elif self.encode_type == 'word2vec':
            return (self.word2vec(train_X, model_path=self.model_path, n_grams=n_grams), train_y), (
            self.word2vec(test_X, model_path=self.model_path, n_grams=n_grams), test_y)
        # elif self.encode_type == 'spacy':
        #     return self.spacy(x), target

    def one_hot(self, x, vocab, n_grams) -> np.ndarray:
        """
        Convert corpus into batch of one-hot vectors.
        :param x:
        :param vocab:
        :param n_grams
        :return:
        """
        _x = np.zeros((x.__len__(), self.max_length, vocab.__len__()))
        for i, item in enumerate(tqdm(x, desc='One Hot Text Encoder')):
            items = TextProcessor.n_gram_split(item, n_grams)
            for j, token in enumerate(items):
                if j >= self.max_length:
                    break
                idx = vocab.index(token)
                _x[i][j][idx] = 1

        return _x

    def bow(self, x, vocab, n_grams) -> np.ndarray:
        """
        Bag-of-Word encoder
        :param x
        :param vocab
        :param n_grams
        :return:
        """
        _x = np.zeros((x.__len__(), vocab.__len__()))
        for i, item in enumerate(tqdm(x, desc='Bag Of Words Text Encoder')):
            items = TextProcessor.n_gram_split(item, n_grams)
            for token in items:
                j = vocab.index(token)
                _x[i][j] = _x[i][j] + 1

        return _x

    def tf_idf(self, x, vocab, n_grams) -> np.ndarray:
        """
        Simple TF-IDF feature
        :param x:
        :param vocab:
        :param n_grams
        :return:
        """
        items = [TextProcessor.n_gram_split(item, n_grams) for item in x]
        appearances_in_doc = {k: 0 for k in vocab}

        for _ in items:
            _set = set()
            for __ in _:
                if __ not in _set:
                    appearances_in_doc[__] += 1
                    _set.add(__)

        import math

        _x = np.zeros((x.__len__(), vocab.__len__()))
        for i, _ in enumerate(tqdm(items, desc='TF-IDF Text Encoder')):
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

    def word2vec(self, x, model_path, n_grams) -> np.ndarray:
        """
        Convert corpus instance into glove
        :param x:
        :param model_path
        :param n_grams
        :return:
        """
        if self.word_vectors is None:
            import gensim
            from distutils.version import LooseVersion

            if LooseVersion(gensim.__version__) >= LooseVersion('1.0.1'):
                from gensim.models import KeyedVectors
                self.word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
            else:
                from gensim.models import Word2Vec
                self.word_vectors = Word2Vec.load_word2vec_format(model_path, binary=True)

        _x = None

        for item in tqdm(x, desc='Word2Vec Text Encoder'):
            __x = None
            items = TextProcessor.n_gram_split(item, n_grams)
            for token in items:
                try:
                    vector = self.word_vectors.get_vector(token)
                    vector = np.expand_dims(vector, axis=0)
                    __x = vector if __x is None else np.concatenate((__x, vector))
                except Exception as ex:
                    continue

            if __x.shape[0] < self.max_length:
                adjust_size = self.max_length - __x.shape[0]
                adjust_array = np.zeros((adjust_size, 400))
                __x = np.concatenate((__x, adjust_array))

            __x = np.expand_dims(__x, axis=0)
            _x = __x if _x is None else np.concatenate((_x, __x))

        return _x

    # def spacy(self, x: Corpus) -> np.ndarray:
    #     """
    #     Using vi-spacy
    #     :param x:
    #     :return:
    #     """
    #     import spacy
    #     nlp = spacy.load('vi_spacy_model')
    #
    #     for item, _ in tqdm(x, desc='Vi-Spacy Text Encoder'):
    #         __x = None
    #         print(item)
