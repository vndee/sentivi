from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import LSTMClassifier
from sentivi.text_processor import TextProcessor

if __name__ == '__main__':
    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation', 'lower'])

    pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=2, max_length=100),
                        TextEncoder(encode_type='word2vec', model_path='./pretrained/wiki.vi.model.bin.gz'),
                        LSTMClassifier(num_labels=2, bidirectional=False, attention=True, device='cuda',
                                       hidden_layers=1))

    train_results = pipeline(train='./data/dev.vi', test='./data/dev_test.vi',
                             num_epochs=3, learning_rate=1e-3)
    print(train_results)

    predict_results = pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                        'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                        'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                        'đẹppppp', 'Son rất đẹp màu xinh lắm'])
    print(predict_results)
    print(f'Decoded results: {pipeline.decode_polarity(predict_results)}')

    pipeline.save(f'./weights/lstm.sentivi')
