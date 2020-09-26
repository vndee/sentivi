from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import NaiveBayesClassifier
from sentivi.text_processor import TextProcessor


if __name__ == '__main__':
    text_processor = TextProcessor()
    text_processor.word_segmentation()
    text_processor.remove_punctuation()
    text_processor.lower()

    pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=2),
                        TextEncoder(encode_type='tf_idf'))
    print(pipeline('./data/dev.vi'))
