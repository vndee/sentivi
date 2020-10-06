import os
import argparse

from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import *
from sentivi.text_processor import TextProcessor

CLASSIFIER = TransformerClassifier


if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser(description='Sentiment Analysis Experiments')
    argument_parser.add_argument('--n_grams', type=int, default=1)
    argument_parser.add_argument('--train_file', type=str, default=os.path.join('data', 'data_done.txt'))
    argument_parser.add_argument('--test_file', type=str, default=os.path.join('data', 'test_data.txt'))
    argument_parser.add_argument('--log', type=str, default=os.path.join('data', 'logs', f'{CLASSIFIER.__name__}.txt'))
    args = argument_parser.parse_args()

    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation'])

    encoding = 'transformer'
    file_writer = open(args.log, 'w+')

    train_pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=args.n_grams, max_length=256),
                              TextEncoder(encode_type=encoding),
                              CLASSIFIER(num_labels=3, language_model_shortcut='vinai/phobert-base', device='cuda'))

    train_results = train_pipeline(train=args.train_file, test=args.test_file, num_epochs=30, batch_size=4)

    print(f'Experiment_{encoding}_{CLASSIFIER.__name__}:\n{train_results}')
    file_writer.write(f'Experiment_{encoding}_{CLASSIFIER.__name__}:\n{train_results}\n')
    file_writer.write('*'*15 + '\n')

    file_writer.close()
