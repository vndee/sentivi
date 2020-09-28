import os
import logging

from typing import Optional
from sentivi.base_model import DataLayer
from sentivi.text_processor import TextProcessor


class Corpus(object):
    def __init__(self,
                 train_file: Optional[str] = None,
                 test_file: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None,
                 max_length: Optional[int] = 256,
                 truncation: Optional[str] = 'head'):
        """
        Text corpus for sentiment analysis
        :param train_file: Path to train text file
        :param test_file: Path to test text file
        :param delimiter: Separator between text and labels
        :param line_separator: Separator between samples.
        :param n_grams: N-grams
        :param text_processor:
        :param max_length:
        """
        super(Corpus, self).__init__()

        if train_file is None:
            raise ValueError('train_file parameter is required.')
        elif not os.path.exists(train_file):
            raise FileNotFoundError(f'Could not found {train_file}.')

        if test_file is None:
            raise ValueError('test_file parameter is required.')
        elif not os.path.exists(test_file):
            raise FileNotFoundError(f'Could not found {test_file}')

        self.__train_file = train_file
        self.__test_file = test_file
        self.__delimiter = delimiter
        self.__line_separator = line_separator
        self.__text_processor = text_processor

        self.__train_sentences = list()
        self.__train_sentiments = list()
        self.__test_sentences = list()
        self.__test_sentiments = list()

        self.vocab = None
        self.labels_set = None
        self.n_grams = n_grams
        self.max_length = max_length
        self.truncation = truncation

        self.build()

    def text_transform(self, text):
        __text = [_x for _x in self.__text_processor(text).split(' ') if _x != '']
        if __text.__len__() > self.max_length:
            if self.truncation == 'head':
                __text = __text[:self.max_length]
            elif self.truncation == 'tail':
                __text = __text[__text.__len__() - self.max_length:]
            else:
                raise ValueError(f'truncation method must be in [head, tail] - not {self.truncation}')
        return ' '.join(__text)

    def build(self):
        """
        Build sentivi.Corpus instance
        :return:
        """
        warehouse = set()
        label_set = set()

        train_file_reader = open(self.__train_file, 'r').read().split(self.__line_separator)
        for line in train_file_reader:
            line = line.split('\n')
            label, text = line[0], ' '.join(line[1:])
            text = self.text_transform(text)
            self.__train_sentences.append(text)
            self.__train_sentiments.append(label)

            if label not in label_set:
                label_set.add(label)

            words = self.__text_processor.n_gram_split(text, self.n_grams)
            for word in words:
                if word not in warehouse:
                    warehouse.add(word)

        test_file_reader = open(self.__test_file, 'r').read().split(self.__line_separator)
        for line in test_file_reader:
            line = line.split('\n')
            label, text = line[0], ' '.join(line[1:])
            text = self.text_transform(text)

            self.__test_sentences.append(text)
            self.__test_sentiments.append(label)

            if label not in label_set:
                label_set.add(label)

            words = self.__text_processor.n_gram_split(text, self.n_grams)
            for word in words:
                if word not in warehouse:
                    warehouse.add(word)

        label_set = list(label_set)
        self.__train_sentiments = [label_set.index(sent) for sent in self.__train_sentiments]
        self.__test_sentiments = [label_set.index(sent) for sent in self.__test_sentiments]
        self.vocab = list(warehouse)

        assert self.__train_sentences.__len__() == self.__train_sentiments.__len__(), ValueError(
            'Index value is out of bound.')
        assert self.__test_sentences.__len__() == self.__test_sentiments.__len__(), ValueError(
            'Index value is out of bound.')

        self.labels_set = label_set
        return self.vocab

    def get_train_set(self):
        return self.__train_sentences, self.__train_sentiments

    def get_test_set(self):
        return self.__test_sentences, self.__test_sentiments


class DataLoader(DataLayer):
    def __init__(self,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None,
                 max_length: Optional[int] = 256):
        super(DataLoader, self).__init__()

        if delimiter is None:
            delimiter = '\n'
            logging.warning(f'Default delimiter will be \'\\n\'')

        if line_separator is None:
            line_separator = '\n\n'
            logging.warning(f'Default line_separator will be \'\\n\\n\'')

        if n_grams is None:
            n_grams = 1
            logging.warning(f'Default n_grams will be 1')

        if max_length is None:
            max_length = 256
            logging.warning(f'Default max_length will be 256')

        self.__delimiter = delimiter
        self.__line_separator = line_separator

        self.max_length = max_length
        self.n_grams = n_grams
        self.text_processor = text_processor
        self.vocab = None
        self.labels_set = None

    def __call__(self, *args, **kwargs):
        assert 'train' in kwargs, ValueError('train parameter is required.')
        assert 'test' in kwargs, ValueError('test parameter is required.')

        corpus = Corpus(train_file=kwargs['train'], test_file=kwargs['test'], delimiter=self.__delimiter,
                        line_separator=self.__line_separator, n_grams=self.n_grams, text_processor=self.text_processor,
                        max_length=self.max_length)
        self.vocab = corpus.vocab
        self.labels_set = corpus.labels_set
        return corpus
