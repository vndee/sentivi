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


class DataLayer(object):
    def __init__(self):
        super(DataLayer, self).__init__()

    def __call__(self, *args, **kwargs):
        pass


class ClassifierLayer(object):
    def __init__(self):
        super(ClassifierLayer, self).__init__()

    def fit(self, *args, **kwargs):
        pass

    def save(self, *args, **kwargs):
        pass


class Pipeline(object):
    def __init__(self, *args):
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

    def __call__(self, x):
        for method in self.apply_layers:
            x = method(x)
        return x
