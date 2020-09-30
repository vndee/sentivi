import os
import logging

from typing import Optional
from sentivi.base_model import DataLayer
from sentivi.text_processor import TextProcessor


class Corpus(object):
    """
    Text corpus for sentiment analysis
    """
    def __init__(self,
                 train_file: Optional[str] = None,
                 test_file: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None,
                 max_length: Optional[int] = None,
                 truncation: Optional[str] = 'head'):
        """
        Initialize Corpus instance

        :param train_file: Path to train text file
        :param test_file: Path to test text file
        :param delimiter: Separator between text and labels
        :param line_separator: Separator between samples.
        :param n_grams: N-grams
        :param text_processor: sentivi.text_processor.TextProcessor instance
        :param max_length: maximum length of input text
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
        """
        Preprocessing raw text

        :param text: raw text
        :return: text
        :rtype: str
        """
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
        Build sentivi.data.data_loader.Corpus instance

        :return: sentivi.data.data_loader.Corpus instance
        :rtype: sentivi.data.data_lodaer.Corpus
        """
        warehouse = set()
        label_set = set()

        train_file_reader = None
        with open(self.__train_file, 'r') as stream:
            train_file_reader = stream.read().split(self.__line_separator)

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

        test_file_reader = None
        with open(self.__test_file, 'r') as stream:
            test_file_reader = stream.read().split(self.__line_separator)

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
        """
        Get training samples

        :return: Input and output of training samples
        :rtype: Tuple[List, List]
        """
        return self.__train_sentences, self.__train_sentiments

    def get_test_set(self):
        """
        Get test samples

        :return: Input and output of test samples
        :rtype: Tuple[List, List]
        """
        return self.__test_sentences, self.__test_sentiments


class DataLoader(DataLayer):
    """
    DataLoader is an inheritance class of DataLayer.
    """
    def __init__(self,
                 delimiter: Optional[str] = '\n',
                 line_separator: Optional[str] = '\n\n',
                 n_grams: Optional[int] = 1,
                 text_processor: Optional[TextProcessor] = None,
                 max_length: Optional[int] = 256):
        """
        :param delimiter: separator between polarity and text
        :param line_separator: separator between samples
        :param n_grams: n-gram(s) use to split, for TextEncoder such as word2vec or transformer, n-gram should be 1
        :param text_processor: sentivi.text_processor.TextProcessor instance
        :param max_length: maximum length of input text
        """
        super(DataLoader, self).__init__()

        self.__delimiter = delimiter
        self.__line_separator = line_separator

        self.max_length = max_length
        self.n_grams = n_grams
        self.text_processor = text_processor
        self.vocab = None
        self.labels_set = None

    def forward(self, *args, **kwargs):
        """
        Execute loading data pipeline

        :param args: arbitrary arguments
        :param kwargs: arbitrary keyword arguments
        :return: loaded data
        :rtype: sentivi.data.data_loader.Corpus
        """
        assert 'train' in kwargs, ValueError('train parameter is required.')
        assert 'test' in kwargs, ValueError('test parameter is required.')

        corpus = Corpus(train_file=kwargs['train'], test_file=kwargs['test'], delimiter=self.__delimiter,
                        line_separator=self.__line_separator, n_grams=self.n_grams, text_processor=self.text_processor,
                        max_length=self.max_length)

        self.vocab = corpus.vocab
        self.labels_set = corpus.labels_set
        return corpus

    __call__ = forward
