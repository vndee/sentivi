import unittest
import numpy as np

from sentivi import Pipeline

from sentivi.data import TextEncoder, DataLoader
from sentivi.text_processor import TextProcessor

from sentivi.classifier import GaussianProcessClassifier


class GaussianProcessTestCase(unittest.TestCase):
    INPUT_FILE = './data/dev.vi'
    OUTPUT_FILE = './data/dev_test.vi'
    SAVED_PATH = './weights/pipeline_test.sentivi'

    W2V_PRETRAINED = './pretrained/wiki.vi.model.bin.gz'

    class EncodingAliases:
        ONE_HOT = 'one-hot'
        BOW = 'bow'
        TF_IDF = 'tf-idf'
        W2V = 'word2vec'
        TRANSFORMER = 'transformer'

    text_processor = TextProcessor(methods=['remove_punctuation', 'word_segmentation'])

    def test_text_processor(self):
        self.assertIsInstance(self.text_processor, TextProcessor)
        self.assertEqual(self.text_processor('Trường đại học,   Tôn Đức Thắng, Hồ; Chí Minh.'),
                         'Trường đại_học Tôn_Đức_Thắng Hồ_Chí_Minh')

    def test_one_hot(self):
        pipeline = Pipeline(DataLoader(text_processor=self.text_processor, n_grams=3),
                            TextEncoder(encode_type=GaussianProcessTestCase.EncodingAliases.ONE_HOT),
                            GaussianProcessClassifier(num_labels=3))
        pipeline(train=GaussianProcessTestCase.INPUT_FILE, test=GaussianProcessTestCase.OUTPUT_FILE)
        pipeline.save(GaussianProcessTestCase.SAVED_PATH)

        _pipeline = Pipeline.load(GaussianProcessTestCase.SAVED_PATH)
        self.assertIsInstance(_pipeline, Pipeline)

        predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi '
                                             '.cần nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                             'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên '
                                             'chuẩn, đẹppppp'])

        self.assertIsInstance(predict_results, np.ndarray)

    def test_bow(self):
        pipeline = Pipeline(DataLoader(text_processor=self.text_processor, n_grams=3),
                            TextEncoder(encode_type=GaussianProcessTestCase.EncodingAliases.BOW),
                            GaussianProcessClassifier(num_labels=3))
        pipeline(train=GaussianProcessTestCase.INPUT_FILE, test=GaussianProcessTestCase.OUTPUT_FILE)
        pipeline.save(GaussianProcessTestCase.SAVED_PATH)

        _pipeline = Pipeline.load(GaussianProcessTestCase.SAVED_PATH)
        self.assertIsInstance(_pipeline, Pipeline)

        predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi '
                                             '.cần nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                             'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên '
                                             'chuẩn, đẹppppp'])

        self.assertIsInstance(predict_results, np.ndarray)

    def test_tf_idf(self):
        pipeline = Pipeline(DataLoader(text_processor=self.text_processor, n_grams=3),
                            TextEncoder(encode_type=GaussianProcessTestCase.EncodingAliases.TF_IDF),
                            GaussianProcessClassifier(num_labels=3))
        pipeline(train=GaussianProcessTestCase.INPUT_FILE, test=GaussianProcessTestCase.OUTPUT_FILE)
        pipeline.save(GaussianProcessTestCase.SAVED_PATH)

        _pipeline = Pipeline.load(GaussianProcessTestCase.SAVED_PATH)
        self.assertIsInstance(_pipeline, Pipeline)

        predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi '
                                             '.cần nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                             'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên '
                                             'chuẩn, đẹppppp'])

        self.assertIsInstance(predict_results, np.ndarray)

    def test_word2vec(self):
        pipeline = Pipeline(DataLoader(text_processor=self.text_processor, n_grams=3),
                            TextEncoder(encode_type=GaussianProcessTestCase.EncodingAliases.W2V,
                                        model_path=GaussianProcessTestCase.W2V_PRETRAINED),
                            GaussianProcessClassifier(num_labels=3))
        pipeline(train=GaussianProcessTestCase.INPUT_FILE, test=GaussianProcessTestCase.OUTPUT_FILE)
        pipeline.save(GaussianProcessTestCase.SAVED_PATH)

        _pipeline = Pipeline.load(GaussianProcessTestCase.SAVED_PATH)
        self.assertIsInstance(_pipeline, Pipeline)

        predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi '
                                             '.cần nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                             'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên '
                                             'chuẩn, đẹppppp'])

        self.assertIsInstance(predict_results, np.ndarray)
