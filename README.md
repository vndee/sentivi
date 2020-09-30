## A Simple Tool For Sentiment Analysis

**Sentivi** - a simple tool for sentiment analysis which is a wrapper of [scikit-learn](https://scikit-learn.org) and
[PyTorch Transformers](https://huggingface.co/transformers/) models (for more special purpose, it is recommend to use native library instead). It is made for easy and faster pipeline to train and evaluate several
classification algorithms.

Documentation: https://sentivi.readthedocs.io/en/latest/index.html

### Classifiers

- [x] Decision Tree
- [x] Gaussian Naive Bayes
- [x] Gaussian Process
- [x] Nearest Centroid
- [x] Support Vector Machine
- [x] Stochastic Gradient Descent
- [ ] Character Convolutional Neural Network
- [x] Multi-Layer Perceptron
- [x] Long Short Term Memory
- [x] Text Convolutional Neural Network
- [x] Transformer
- [ ] Ensemble
- [ ] Lexicon-based 

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

train_results = pipeline(train='./data/dev.vi', test='./data/dev_test.vi')
print(train_results)

pipeline.save('./weights/pipeline.sentivi')
_pipeline = Pipeline.load('./weights/pipeline.sentivi')

predict_results = _pipeline.predict(['hàng ok đầu tuýp có một số không vừa ốc siết. chỉ được một số đầu thôi .cần '
                                    'nhất đầu tuýp 14 mà không có. không đạt yêu cầu của mình sử dụng',
                                    'Son đẹpppp, mùi hương vali thơm nhưng hơi nồng, chất son mịn, màu lên chuẩn, '
                                    'đẹppppp'])
print(predict_results)
print(f'Decoded results: {_pipeline.decode_polarity(predict_results)}')
```
Take a look at more examples in [example/](https://github.com/vndee/sentivi/tree/master/example).

### Future Releases

- Lexicon-based
- CharCNN
- Ensemble learning methods
- Model serving (Back-end and Front-end)
