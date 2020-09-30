FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

COPY . /app

ENV PYTHONPATH=/app
ENV APP_MODULE=test.serving:server
ENV WORKERS_PER_CORE=0.75
ENV MAX_WORKERS=6
ENV HOST=0.0.0.0
ENV PORT=80

RUN pip install -r requirements.txt