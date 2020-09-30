Text Encoder
***************************
``TextEncoder`` is a class that receives pre-processed data from ``sentivi.data.DataLoader``, its responsibility is
to provide appropriate data to the respective classifications.

.. code-block:: python

    text_encoder = TextEncoder('one-hot') # ['one-hot', 'word2vec', 'bow', 'tf-idf', 'transformer']


.. autoclass:: sentivi.data.TextEncoder
    :members:
