import os
import logging

from typing import Optional
from sentivi import DataLayer
from sentivi.text_processor import TextProcessor


class Corpus(object):
    def __init__(self,
                 train_file: Optional[str] = None,
                 test_file: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None):
        """
        Text corpus for sentiment analysis
        :param train_file: Path to train text file
        :param test_file: Path to test text file
        :param delimiter: Separator between text and labels
        :param line_separator: Separator between samples.
        :param n_grams: N-grams
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
        self.n_grams = n_grams
        self.__text_processor = text_processor

        self.vocab = None
        self.__train_sentences = list()
        self.__train_sentiments = list()
        self.__test_sentences = list()
        self.__test_sentiments = list()

        self.build()

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
            text = ' '.join([_x for _x in self.__text_processor(text).split(' ') if _x != ''])

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
            text = ' '.join([_x for _x in self.__text_processor(text).split(' ') if _x != ''])

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

    def get_train_set(self):
        return self.__train_sentences, self.__train_sentiments

    def get_test_set(self):
        return self.__test_sentences, self.__test_sentiments

    def export_vocab(self, file_path):
        file_writer = open(file_path, mode='w+')
        for token in self.vocab:
            file_writer.write(token + '\n')
        file_writer.close()
        print(f'Exported vocabulary to {file_path}')


class DataLoader(DataLayer):
    def __init__(self,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None):
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

        self.__delimiter = delimiter
        self.__line_separator = line_separator
        self.__n_grams = n_grams
        self.__text_processor = text_processor

    def __call__(self, *args, **kwargs):
        assert 'train' in kwargs, ValueError('train parameter is required.')
        assert 'test' in kwargs, ValueError('test parameter is required.')

        return Corpus(train_file=kwargs['train'], test_file=kwargs['test'], delimiter=self.__delimiter,
                      line_separator=self.__line_separator, n_grams=self.__n_grams,
                      text_processor=self.__text_processor)
