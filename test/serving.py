from sentivi import Pipeline, RESTServiceGateway

pipeline = Pipeline.load('./weights/pipeline.sentivi')
server = RESTServiceGateway(pipeline).get_server()

# docker run -d --name sentivi_serving_02 -p 8000:80 --env APP_MODULE=test.serving:server --env WORKERS_PER_CORE=0.75
# --env MAX_WORKERS=6 --env HOST=0.0.0.0 --env PORT=80 --env PYTHONPATH=/app -v `pwd`:/app
# tiangolo/uvicorn-gunicorn-fastapi:python3.7 pip install -r requirements.txt
