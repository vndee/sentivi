Transformer Classifier
***********************

``TransformerClassifier`` base on `transformers <https://huggingface.co/transformers>`_ library. This is a wrapper of
``transformers.AutoModelForSequenceClassification``, language model should be one of shortcut in `transformers
pretrained models <https://huggingface.co/transformers/pretrained_models.html>`_ or using one in ``['vinai/phobert-base',
'vinai/phobert-large']``

.. code-block:: python

    TransformerClassifier(num_labels=3, language_model_shortcut='vinai/phobert-base', device='cuda')

.. autoclass:: sentivi.classifier.TransformerClassifier
    :members:
