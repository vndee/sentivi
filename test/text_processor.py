from sentivi.text_processor import TextProcessor


if __name__ == '__main__':
    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation', 'lower'])
    print(text_processor('ASLKn lajdf;.;'))
