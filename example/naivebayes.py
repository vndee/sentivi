from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import NaiveBayesClassifier
from sentivi.text_processor import TextProcessor


if __name__ == '__main__':
    text_processor = TextProcessor()
    text_processor.word_segmentation()
    text_processor.remove_punctuation()
    text_processor.lower()

    pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=1),
                        TextEncoder(encode_type='word2vec', model_path='./pretrained/wiki.vi.model.bin.gz'))
                        # NaiveBayesClassifier(num_labels=3))
    (train_X, train_y), (test_X, test_y) = pipeline(train='./data/dev.vi', test='./data/dev_test.vi')
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
