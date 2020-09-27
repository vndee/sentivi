from typing import Optional
from sentivi.data import DataLoader


class Pipeline(object):
    def __init__(self, *args):
        super(Pipeline, self).__init__()
        self.apply_layers = list()
        for method in args:
            self.apply_layers.append(method)

    def __call__(self, *args, **kwargs):
        x = None
        for method in self.apply_layers:
            x = method(x, *args, **kwargs)
        return x

    def predict(self, x: Optional[list], *args, **kwargs):
        n_grams, vocab = None, None
        for method in self.apply_layers:
            if isinstance(method, DataLoader):
                n_grams, vocab = method.n_grams, method.vocab
                continue
            x = method.predict(x, vocab=vocab, n_grams=n_grams, *args, **kwargs)
            print(x)
        return x
