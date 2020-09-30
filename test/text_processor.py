import unittest
from sentivi.text_processor import TextProcessor


if __name__ == '__main__':
    text_processor = TextProcessor(methods=['remove_punctuation', 'word_segmentation'])
    print(text_processor('Trường đại học,   Tôn Đức Thắng, Hồ; Chí Minh.'))
    print(TextProcessor.n_gram_split('bài tập phân tích cảm xúc', 3))
