from sklearn.neural_network import MLPClassifier as _MLPClassifier
from sentivi.classifier.sklearn_clf import ScikitLearnClassifier


class MLPClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(MLPClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)

        mlp_params = {
            'hidden_layer_sizes': kwargs.get('hidden_layer_sizes', (100,)),
            'activation': kwargs.get('activation', 'relu'),
            'solver': kwargs.get('solver', 'adam'),
            'alpha': kwargs.get('alpha', 0.0001),
            'batch_size': kwargs.get('batch_size', 'auto'),
            'learning_rate': kwargs.get('learning_rate', 'constant'),
            'learning_rate_init': kwargs.get('learning_rate_init', 0.001),
            'power_t': kwargs.get('power_t', 0.5),
            'max_iter': kwargs.get('max_iter', 200),
            'shuffle': kwargs.get('shuffle', True),
            'random_state': kwargs.get('random_state', None),
            'tol': kwargs.get('tol', 0.0001),
            'verbose': kwargs.get('verbose', True),
            'warm_start': kwargs.get('warm_start', False),
            'momentum': kwargs.get('momentum', 0.9),
            'nesterovs_momentum': kwargs.get('nesterovs_momentum', True),
            'early_stopping': kwargs.get('early_stopping', False),
            'validation_fraction': kwargs.get('validation_fraction', 0.1),
            'beta_1': kwargs.get('beta_1', 0.9),
            'beta_2': kwargs.get('beta_2', 0.999),
            'epsilon': kwargs.get('epsilon', 1e-08),
            'n_iter_no_change': kwargs.get('n_iter_no_change', 10),
            'max_fun': 15000
        }

        self.clf = _MLPClassifier(**mlp_params)
