## A Simple Tool For Sentiment Analysis

**Sentivi** - a simple tool for sentiment analysis which is a wrapper of [scikit-learn](https://scikit-learn.org) and
[PyTorch Transformers](https://huggingface.co/transformers/) models (for more specific purpose, it is recommend to use native library instead). It is made for easy and faster pipeline to train and evaluate several
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

### Text Encoders

- [x] One-hot
- [x] Bag of Words
- [x] Term Frequency - Inverse Document Frequency
- [x] Word2Vec
- [x] Transformer Tokenizer (for Transformer classifier only)
- [ ] WordPiece
- [ ] SentencePiece

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

### Pipeline Serving

Sentivi use [FastAPI](https://fastapi.tiangolo.com/) to serving pipeline. Simply run a web service as follows:

```python
# serving.py
from sentivi import Pipeline, RESTServiceGateway

pipeline = Pipeline.load('./weights/pipeline.sentivi')
server = RESTServiceGateway(pipeline).get_server()

```

```bash
# pip install uvicorn python-multipart
uvicorn serving:server --host 127.0.0.1 --port 8000
```
Access Swagger at http://127.0.0.1:8000/docs or Redoc http://127.0.0.1:8000/redoc. For example, you can use
[curl](https://curl.haxx.se/) to send post requests:

```bash
curl --location --request POST 'http://127.0.0.1:8000/get_sentiment/' \
     --form 'text=Son đẹpppp, mùi hương vali thơm nhưng hơi nồng'

# response
{ "polarity": 2, "label": "#POS" }
```

#### Deploy using Docker
```dockerfile
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app

ENV PYTHONPATH=/app
ENV APP_MODULE=serving:server
ENV WORKERS_PER_CORE=0.75
ENV MAX_WORKERS=6
ENV HOST=0.0.0.0
ENV PORT=80

RUN pip install -r requirements.txt
```

```bash
docker build -t sentivi .
docker run -d -p 8000:80 sentivi
```

### Future Releases

- Lexicon-based
- CharCNN
- Ensemble learning methods
- Model serving (Back-end and Front-end)
