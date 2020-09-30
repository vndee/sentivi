from sentivi import Pipeline, RESTServiceGateway

pipeline = Pipeline.load('./weights/pipeline.sentivi')
server = RESTServiceGateway(pipeline).get_server()
