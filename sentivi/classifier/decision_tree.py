from sentivi.classifier.sklearn_clf import ScikitLearnClassifier
from sklearn.tree import DecisionTreeClassifier as _DecisionTreeClassifier


class DecisionTreeClassifier(ScikitLearnClassifier):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(DecisionTreeClassifier, self).__init__(num_labels=num_labels, *args, **kwargs)

        decision_tree_params = {
            'criterion': kwargs.get('criterion', 'gini'),
            'splitter': kwargs.get('splitter', 'best'),
            'max_depth': kwargs.get('max_depth', None),
            'min_samples_split': kwargs.get('min_samples_split', 2),
            'min_samples_leaf': kwargs.get('min_samples_leaf', 1),
            'min_weight_fraction_leaf': kwargs.get('min_weight_fraction_leaf', 0.0),
            'max_features': kwargs.get('max_features', None),
            'random_state': kwargs.get('random_state', None),
            'max_leaf_nodes': kwargs.get('max_leaf_nodes', None),
            'min_impurity_decrease': kwargs.get('min_impurity_decrease', 0.0),
            'min_impurity_split': kwargs.get('min_impurity_split', None),
            'class_weight': kwargs.get('class_weight', None),
            'presort': kwargs.get('presort', 'deprecated'),
            'ccp_alpha': kwargs.get('ccp_alpha', 0.0)
        }

        self.clf = _DecisionTreeClassifier(**decision_tree_params)
