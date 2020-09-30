Pipeline
**********

Pipeline is a sequence of callable layer (``DataLayer``, ``ClassifierLayer``). These layers will be executed with given
input (text file) sequentially. Output of the pipeline is the output of last executed layer.

Pipeline can be initialized by default constructor, callable layer can be passed through pipeline in initialization once
or use append method.

For example:

.. code-block:: python

    from sentivi import Pipeline
    from sentivi.data import DataLoader, TextEncoder
    from sentivi.classifier import SVMClassifier
    from sentivi.text_processor import TextProcessor

    text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation', 'lower'])

    pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=3),
                        TextEncoder(encode_type='one-hot'),
                        SVMClassifier(num_labels=3))

or

.. code-block:: python

    pipeline = Pipeline()
    pipeline.append(DataLoader(text_processor=text_processor, n_grams=3))
    pipeline.append(TextEncoder(encode_type='one-hot'))
    pipeline.append(SVMClassifier(num_labels=3))

Executing pipeline with given corpus (text file). By default text file should be in our format, double newline character
``(\n\n)`` is the separated symbol of training samples:

.. code-block::

    #corpus.txt
    polarity_01
    sentence_01

    polarity_02
    sentence_02

Pipeline also accept arbitrary keyword arguments when executed function is call, these arguments is passed through
executed functions of each layer. Training results will be represented as text in the form of
``sklearn.metrics.classification_report``.

.. code-block:: python

    results = pipeline(train='train.txt', test='test.txt')

.. code-block::

    #results

    Training classifier...
    Testing classifier...
    Saved classifier model to ./weights/svm.sentivi

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

Predict polarity with given texts:

.. code-block:: python

    predict_results = pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                        'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                        'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                        'đẹppppp'])
    print(predict_results)
    print(f'Decoded results: {pipeline.decode_polarity(predict_results)}')

.. code-block::

    [2 1]
    Decoded results: ['#NEG', '#POS']

For persistency, pipe can be save and load later:

.. code-block:: python

    pipeline.save('./weights/pipeline.sentivi')
    _pipeline = Pipeline.load('./weights/pipeline.sentivi')

    predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                        'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                        'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                        'đẹppppp'])
    print(predict_results)
    print(f'Decoded results: {_pipeline.decode_polarity(predict_results)}')

.. autoclass:: sentivi.Pipeline
    :members:
