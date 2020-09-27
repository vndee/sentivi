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
                n_grams, vocab, text_processor = method.n_grams, method.vocab, method.text_processor
                x = [' '.join([_text for _text in text_processor(text).split(' ') if _text != '']) for text in x]
                continue
            x = method.predict(x, vocab=vocab, n_grams=n_grams, *args, **kwargs)
        return x

    def decode_polarity(self, x: Optional[list]):
        for method in self.apply_layers:
            if isinstance(method, DataLoader):
                labels_set = method.labels_set
                results = [labels_set[idx] for idx in x]
                return results
