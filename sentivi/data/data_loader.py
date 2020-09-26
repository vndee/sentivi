import os
import logging

from typing import Optional
from sentivi import DataLayer
from sentivi.text_processor import TextProcessor


class Corpus(object):
    def __init__(self, source_file: Optional[str] = None,
                 delimiter: Optional[str] = None,
                 line_separator: Optional[str] = None,
                 n_grams: Optional[int] = None,
                 text_processor: Optional[TextProcessor] = None):
        """
        Text corpus for sentiment analysis
        :param source_file: Path to text file contains your corpus
        :param delimiter: Separator between text and labels
        :param line_separator: Separator between samples.
        :param n_grams: N-grams
        """
        super(Corpus, self).__init__()

        if source_file is None:
            raise ValueError('source_file parameter is required.')
        elif not os.path.exists(source_file):
            raise FileNotFoundError(f'Could not found {source_file}.')

        self.__source_file = source_file
        self.__delimiter = delimiter
        self.__line_separator = line_separator
        self.n_grams = n_grams
        self.__text_processor = text_processor

        self.vocab = None
        self.__sentences = list()
        self.__sentiments = list()

        self.build()

    def build(self):
        """
        Build sentivi.Corpus instance
        :return:
        """
        file_reader = open(self.__source_file, 'r').read().split(self.__line_separator)
        warehouse = set()
        label_set = set()

        for line in file_reader:
            line = line.split('\n')
            label, text = line[0], ' '.join(line[1:])
            text = ' '.join([_x for _x in self.__text_processor(text).split(' ') if _x != ''])

            self.__sentences.append(text)
            self.__sentiments.append(label)

            if label not in label_set:
                label_set.add(label)

            words = self.__text_processor.n_gram_split(text, self.n_grams)
            for word in words:
                if word not in warehouse:
                    warehouse.add(word)

        label_set = list(label_set)
        self.__sentiments = [label_set.index(sent) for sent in self.__sentiments]
        self.vocab = list(warehouse)

        assert self.__sentences.__len__() == self.__sentiments.__len__(), ValueError('Index value is out of bound.')

    def __getitem__(self, item):
        return self.__sentences[item], self.__sentiments[item]

    def __len__(self):
        return self.__sentences.__len__()

    def export_vocab(self, file_path):
        file_writer = open(file_path, mode='w+')
        for token in self.__vocab:
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

    def __call__(self, source_file):
        return Corpus(source_file=source_file, delimiter=self.__delimiter, line_separator=self.__line_separator,
                      n_grams=self.__n_grams, text_processor=self.__text_processor)
