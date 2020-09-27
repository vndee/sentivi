from sklearn.svm import SVC
from sentivi.classifier.sklearn_clf import ScikitLearnClassifier


class SVMClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(SVMClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)
        svc_params = {
            'C': kwargs.get('C', 1.0),
            'kernel': kwargs.get('kernel', 'rbf'),
            'degree': kwargs.get('degree', 3),
            'gamma': kwargs.get('gamma', 'scale'),
            'coef0': kwargs.get('coef0', 0.0),
            'shrinking': kwargs.get('shrinking', True),
            'probability': kwargs.get('probability', False),
            'tol': kwargs.get('tol', 1e-3),
            'cache_size': kwargs.get('cache_size', 200),
            'class_weight': kwargs.get('class_weight', None),
            'verbose': kwargs.get('verbose', False),
            'max_iter': kwargs.get('max_iter', -1),
            'decision_function_shape': kwargs.get('decision_function_shape', 'ovr'),
            'break_ties': kwargs.get('break_ties', False),
            'random_state': kwargs.get('random_state', None)
        }

        self.clf = SVC(**svc_params)
