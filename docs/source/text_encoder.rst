Text Encoder
***************************
``TextEncoder`` is a class that receives pre-processed data from ``sentivi.data.DataLoader``, its responsibility is
to provide appropriate data to the respective classifications.

.. code-block:: python

    text_encoder = TextEncoder('one-hot') # ['one-hot', 'word2vec', 'bow', 'tf-idf', 'transformer']

One-hot Encoding
    The simplest encoding type of ``TextEncoder``, each token will be represented as a one-hot vector. These vectors
    indicate the look-up index of given token in corpus vocabulary.
    For example, ``vocab = ['I', 'am', 'a', 'student']``:

        - ``one-hot('I') = [1, 0, 0, 0]``
        - ``one-hot('am') = [0, 1, 0, 0]``
        - ``one-hot('a') = [0, 0, 1, 0]``
        - ``one-hot('student') = [0, 0, 0, 1]``

Bag-of-Words
    A bag-of-words is a representation of text that describes the occurrence of words within a document. It involves
    two things: A vocabulary of known words. A measure of the presence of known words. More detail:
    https://machinelearningmastery.com/gentle-introduction-bag-words-model/

Term Frequency - Inverse Document Frequency
    ``tf–idf`` or TFIDF, short for term frequency–inverse document frequency, is a numerical statistic that is intended to
    reflect how important a word is to a document in a collection or corpus. This ``tf-idf`` version implemented in
    ``TextEncoder`` is logarithmically scaled version. More detail: https://en.wikipedia.org/wiki/Tf%E2%80%93idf

Word2Vec
    Word2vec (`Mikolov et al, 2013 <https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf>`_)
    is a method to efficiently create word embeddings using distributed representation method. This implementation using
    `gensim <https://pypi.org/project/gensim/>`_, it is required ``model_path`` argument in initialization stage. For vietnamese,
    word2vec model should be downloaded from https://github.com/sonvx/word2vecVN

    .. code-block:: python

        text_encoder = TextEncoder(encode_type='word2vec', model_path='./pretrained/wiki.vi.model.bin.gz')

Transformer
    Transformer text encoder is equivalent to ``transformer.AutoTokenizer`` from https://huggingface.co/transformers


.. autoclass:: sentivi.data.TextEncoder
    :members:
