Text Processor
****************************

Sentivi provides a simple text processor layer base on regular expression. ``TextProcessor`` must be defined as a attribute
of ``DataLoader`` layer, it is a required parameter.

List of pre-built methods can be initialized as follows:

.. code-block:: python

    text_processor = TextProcessor(methods=['remove_punctuation', 'word_segmentation'])

    # or add methods sequentially
    text_processor = TextProcessor()
    text_processor.remove_punctuation()
    text_processor.word_segmentation()

    text_processor('Trường đại học,   Tôn Đức Thắng, Hồ; Chí Minh.')

Result:

.. code-block::

    Trường đại_học Tôn_Đức_Thắng Hồ_Chí_Minh

You can also add more regex pattern:

.. code-block:: python

    text_processor.add_pattern(r'[0-9]', '')

Or you can add your own method, use-defined method should be a lambda function.

.. code-block:: python

    text_processor.add_method(lambda x: x.strip())

Split n-grams example:

.. code-block:: python

    TextProcessor.n_gram_split('bài tập phân tích cảm xúc', n_grams=3)

.. code-block::

    ['bài tập phân', 'tập phân tích', 'phân tích cảm', 'tích cảm xúc']

.. autoclass:: sentivi.text_processor.TextProcessor
    :members:
