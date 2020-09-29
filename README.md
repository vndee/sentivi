## A Simple Tool For Sentiment Analysis

**Sentivi** - a simple tool for sentiment analysis which is a wrapper of [scikit-learn](https://scikit-learn.org) and
[PyTorch](https://pytorch.org/) models. It is made for easy and faster pipeline to train and evaluate several
classification algorithms.

### Install
- Install legacy version from PyPI:
    ```bash
    pip install sentivi
    ```

- Install latest version from source:
    ```bash
    git clone https://github.com/vndee/sentivi
    cd sentivi
    pip install .
    ```

### Example

```python
from sentivi import Pipeline
from sentivi.data import DataLoader, TextEncoder
from sentivi.classifier import SVMClassifier
from sentivi.text_processor import TextProcessor

text_processor = TextProcessor(methods=['word_segmentation', 'remove_punctuation', 'lower'])

pipeline = Pipeline(DataLoader(text_processor=text_processor, n_grams=3),
                    TextEncoder(encode_type='one-hot'),
                    SVMClassifier(num_labels=3))

train_results = pipeline(train='./data/dev.vi', test='./data/dev_test.vi',
                         save_path='./weights/svm.sentivi')
print(train_results)

predict_results = pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết.'
                                    'chỉ được một số đầu thôi .cần nhất đầu tuýp 14'
                                    'mà không có. không đạt yêu cầu của mình sử dụng',
                                    'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng,'
                                    'chất son mịn, màu lên chuẩn, đẹppppp'])
print(predict_results)
print(f'Decoded results: {pipeline.decode_polarity(predict_results)}')
```
Take a look at more examples in [example/](https://github.com/vndee/sentivi/tree/master/example).