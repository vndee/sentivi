from sentivi.pipeline import Pipeline


if __name__ == '__main__':
    pipeline = Pipeline.load('./weights/pipeline.sentivi')

    train_results = pipeline(train='./data/dev.vi', test='./data/dev_test.vi')
    print(train_results)

    predict_results = pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                        'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                        'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                        'đẹppppp'])
    print(predict_results)
    print(f'Decoded results: {pipeline.decode_polarity(predict_results)}')

