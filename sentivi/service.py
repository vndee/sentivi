import sentivi
from typing import Optional
from fastapi import FastAPI
from pydantic import BaseModel
from sentivi import Pipeline
from fastapi import Form
from fastapi.responses import HTMLResponse


class ResponseModel(BaseModel):
    polarity: int
    label: str

    class Config:
        schema_extra = {
            'example': [
                {
                    'polarity': 0,
                    'label': '#NEG',
                },
                {
                    'polarity': 1,
                    'label': '#NEU',
                },
                {
                    'polarity': 2,
                    'label': '#POS'
                }
            ]
        }


class RESTServiceGateway:
    server = FastAPI(
        title='Sentivi Web Services',
        description='A simple tool for sentiment analysis',
        version=sentivi.__version__
    )

    pipeline: Optional[Pipeline] = None

    tags_metadata = [
        {
            'name': 'Predictor',
            'description': 'Sentiment Predictor'
        }
    ]

    response_models = {
        'foo': {
            'polarity': 'Numeric polarity',
            'label': 'Label from piplines\' label set'
        }
    }

    def __init__(self, pipeline: Optional[Pipeline], *args, **kwargs):
        super(RESTServiceGateway, self).__init__(*args, **kwargs)

        RESTServiceGateway.pipeline = pipeline
        print('Initialized REST Service Gateway')

    @staticmethod
    @server.get('/', response_class=HTMLResponse)
    async def index():
        return '<h1>Hello, World</h1>'

    @staticmethod
    @server.post('/get_sentiment/', tags=['Predictor'], response_model=ResponseModel)
    async def get_sentiment(text: str = Form(...,
                                             title='Input text',
                                             description='Input text for sentiment analysis')):
        """
        POST method

        :param text:
        :return:
        """
        try:
            polarities = RESTServiceGateway.pipeline.predict([text])
            polarity = polarities[0]
            label = RESTServiceGateway.pipeline.decode_polarity(polarities)[0]
        except Exception as ex:
            polarity = -1
            label = 'ERROR'
            print(f'Error has occurred!!!\n{ex}')

        return {
            'polarity': polarity,
            'label': label
        }

    @staticmethod
    def get_server():
        return RESTServiceGateway.server
