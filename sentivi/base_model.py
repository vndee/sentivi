class SentimentModel(object):
    def __init__(self):
        super(SentimentModel, self).__init__()

    def __call__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass


class TransformationLayer(object):
    def __init__(self):
        super(TransformationLayer, self).__init__()

    def __call__(self, *args, **kwargs):
        pass


class Sequential(object):
    def __init__(self, *args):
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

    def __call__(self, x):
        for method in self.apply_layers:
            x = method(x)
        return x
