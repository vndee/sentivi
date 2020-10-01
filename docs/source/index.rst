Sentivi
*********
A simple tool for sentiment analysis which is a wrapper of `scikit-learn <https://scikit-learn.org>`_ and
`PyTorch Transformers <https://huggingface.co/transformers/>`_ models (for more specific purpose, it is recommend to use native library instead). It is made for easy and faster pipeline to train and evaluate several
classification algorithms.


- Install legacy version from PyPI:

.. code-block::

    pip install sentivi

- Install latest version from source:

.. code-block::

    git clone https://github.com/vndee/sentivi
    cd sentivi
    pip install .

Example:
--------------
.. code-block:: python
    :emphasize-lines: 3,5

    from sentivi import Pipeline
    from sentivi.data import DataLoader, TextEncoder
    from sentivi.classifier import SVMClassifier
    from sentivi.text_processor import TextProcessor

    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation', 'lower'])

    pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=3),
                        TextEncoder(encode_type='one-hot'),
                        SVMClassifier(num_labels=3))

    train_results = pipeline(train='./train.txt', test='./test.txt')
    print(train_results)

    pipeline.save('./weights/pipeline.sentivi')
    _pipeline = Pipeline.load('./weights/pipeline.sentivi')

    predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                        'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                        'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                        'đẹppppp'])
    print(predict_results)
    print(f'Decoded results: {_pipeline.decode_polarity(predict_results)}')


Console output:

.. code-block::

    One Hot Text Encoder: 100%|██████████| 6/6 [00:00<00:00, 11602.50it/s]
    One Hot Text Encoder: 100%|██████████| 2/2 [00:00<00:00, 4966.61it/s]
    Input features view be flatten into np.ndarray(6, 35328) for scikit-learn classifier.
    Training classifier...
    Testing classifier...
    Training results:
                  precision    recall  f1-score   support

               0       1.00      0.00      0.00         1
               1       0.75      1.00      0.86         3
               2       1.00      1.00      1.00         2

        accuracy                           0.83         6
       macro avg       0.92      0.67      0.62         6
    weighted avg       0.88      0.83      0.76         6

    Test results:
                  precision    recall  f1-score   support

               1       1.00      1.00      1.00         1
               2       1.00      1.00      1.00         1

        accuracy                           1.00         2
       macro avg       1.00      1.00      1.00         2
    weighted avg       1.00      1.00      1.00         2


    Saved model to ./weights/pipeline.sentivi
    Loaded model from ./weights/pipeline.sentivi
    Input features view be flatten into np.ndarray(2, 35328) for scikit-learn classifier.
    [2 1]
    Decoded results: ['#NEG', '#POS']
    One Hot Text Encoder: 100%|██████████| 2/2 [00:00<00:00, 10796.15it/s]

.. toctree::
    :maxdepth: 2
    :caption: Contents:

    pipeline
    text_processor
    data_loader
    text_encoder
    sklearn_classifier
    neural_network_classifier
    transformer_classifier
    ensemble_learning
    serving
