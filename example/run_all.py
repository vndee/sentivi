import os
import time
import argparse

from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import DecisionTreeClassifier
from sentivi.text_processor import TextProcessor

ENCODING_TYPE = ['one-hot', 'bow', 'tf-idf', 'word2vec']

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Sentiment Analysis Experiments')
    argument_parser.add_argument('--n_grams', type=int, default=1)
    argument_parser.add_argument('--train_file', type=str, default=os.path.join('data', 'data_done.txt'))
    argument_parser.add_argument('--test_file', type=str, default=os.path.join('data', 'test_data.txt'))
    argument_parser.add_argument('--log', type=str, default=os.path.join('data', 'logs', 'decision_tree.txt'))
    args = argument_parser.parse_args()

    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation'])

    file_writer = open(args.log, 'w+')
    for encoding in ENCODING_TYPE:
        train_pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=args.n_grams),
                                  TextEncoder(encode_type=encoding, model_path='./pretrained/wiki.vi.model.bin.gz'),
                                  DecisionTreeClassifier(num_labels=3))

        train_results = train_pipeline(train=args.train_file, test=args.test_file)

        print(f'Experiment_{encoding}_DecisionTreeClassifier:\n{train_results}')
        file_writer.write(f'Experiment_{encoding}_DecisionTreeClassifier:\n{train_results}\n')
        file_writer.write('*'*15 + '\n')

    file_writer.close()
