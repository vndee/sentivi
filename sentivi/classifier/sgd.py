from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.linear_model import SGDClassifier as _SGDClassifier


class SGDClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(SGDClassifier, self).__init__(num_labels, *args, **kwargs)

        sgd_params = {
            'loss': kwargs.get('loss', 'hinge'),
            'penalty': kwargs.get('penalty', 'l2'),
            'alpha': kwargs.get('alpha', 0.0001),
            'l1_ratio': kwargs.get('l1_ratio', 0.15),
            'fit_intercept': kwargs.get('fit_intercept', True),
            'max_iter': kwargs.get('max_iter', 1000),
            'tol': kwargs.get('tol', 0.001),
            'shuffle': kwargs.get('shuffle', True),
            'verbose': kwargs.get('verbose', True),
            'epsilon': kwargs.get('epsilon', 0.1),
            'n_jobs': kwargs.get('n_jobs', None),
            'random_state': kwargs.get('random_state', None),
            'learning_rate': kwargs.get('learning_rate', 'optimal'),
            'eta0': kwargs.get('eta0', 0.0),
            'power_t': kwargs.get('power_t', 0.5),
            'early_stopping': kwargs.get('early_stopping', False),
            'validation_fraction': kwargs.get('validation_fraction', 0.1),
            'n_iter_no_change': kwargs.get('n_iter_no_change', 5),
            'class_weight': kwargs.get('class_weight', None),
            'warm_start': kwargs.get('warm_start', False),
            'average': kwargs.get('average', False)
        }

        self.clf = _SGDClassifier(**sgd_params)
