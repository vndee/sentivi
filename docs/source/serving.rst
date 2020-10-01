Serving
************

Sentivi use [FastAPI](https://fastapi.tiangolo.com/) to serving pipeline. Simply run a web service as follows:

.. code-block:: python

    # serving.py
    from sentivi import Pipeline, RESTServiceGateway

    pipeline = Pipeline.load('./weights/pipeline.sentivi')
    server = RESTServiceGateway(pipeline).get_server()


.. code-block::

    # pip install uvicorn python-multipart
    uvicorn serving:server --host 127.0.0.1 --port 8000

Access Swagger at http://127.0.0.1:8000/docs or Redoc http://127.0.0.1:8000/redoc. For example, you can use
[curl](https://curl.haxx.se/) to send post requests:

.. code-block::

    curl --location --request POST 'http://127.0.0.1:8000/get_sentiment/' \
         --form 'text=Son đẹpppp, mùi hương vali thơm nhưng hơi nồng'

    # response
    { "polarity": 2, "label": "#POS" }

#### Deploy using Docker
.. code-block:: dockerfile

    FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

    COPY . /app

    ENV PYTHONPATH=/app
    ENV APP_MODULE=serving:server
    ENV WORKERS_PER_CORE=0.75
    ENV MAX_WORKERS=6
    ENV HOST=0.0.0.0
    ENV PORT=80

    RUN pip install -r requirements.txt

.. code-block::

    docker build -t sentivi .
    docker run -d -p 8000:80 sentivi
