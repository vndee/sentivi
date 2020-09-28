from sentivi.base_model import ClassifierLayer
from sklearn.metrics import classification_report


class ScikitLearnClassifier(ClassifierLayer):
    def __init__(self, num_labels: int = 3, *args, **kwargs):
        super(ScikitLearnClassifier, self).__init__()

        self.num_labels = num_labels
        self.clf = None

    def __call__(self, data, *args, **kwargs):
        (train_X, train_y), (test_X, test_y) = data
        assert train_X.shape[1:] == test_X.shape[1:], ValueError(
            f'Number of train features must be equal to test features: {train_X.shape[1:]} != {test_X.shape[1:]}')

        if train_X.shape.__len__() > 2:
            flatten_dim = 1
            for x in train_X.shape[1:]:
                flatten_dim *= x

            print(f'Input features view be flatten into np.ndarray({train_X.shape[0]}, {flatten_dim}) for '
                  f'scikit-learn classifier.')
            train_X = train_X.reshape((train_X.shape[0], flatten_dim))
            test_X = test_X.reshape((test_X.shape[0], flatten_dim))

        # training
        print(f'Training classifier...')
        self.clf.fit(train_X, train_y)
        predicts = self.clf.predict(train_X)
        train_report = classification_report(train_y, predicts, zero_division=1)
        results = f'Training results:\n{train_report}\n'

        # testing
        print(f'Testing classifier...')
        predicts = self.clf.predict(test_X)
        test_report = classification_report(test_y, predicts, zero_division=1)
        results += f'Test results:\n{test_report}\n'

        if 'save_path' in kwargs:
            self.save(kwargs['save_path'])

        return results

    def predict(self, x, *args, **kwargs):
        """
        Predict polarities given sentences
        :param x:
        :param args:
        :param kwargs:
        :return:
        """
        if x.shape.__len__() > 2:
            flatten_dim = 1
            for _x in x.shape[1:]:
                flatten_dim *= _x

            print(f'Input features view be flatten into np.ndarray({x.shape[0]}, {flatten_dim}) for '
                  f'scikit-learn classifier.')
            x = x.reshape((x.shape[0], flatten_dim))
        return self.clf.predict(x)

    def save(self, save_path, *args, **kwargs):
        """
        Dump model to disk
        :param save_path:
        :param args:
        :param kwargs:
        :return:
        """
        import pickle
        with open(save_path, 'wb') as file:
            pickle.dump(self.clf, file)
            print(f'Saved classifier model to {save_path}')

    def load(self, model_path, *args, **kwargs):
        """
        Load model from disk
        :param model_path:
        :param args:
        :param kwargs:
        :return:
        """
        import pickle
        with open(model_path, 'rb') as file:
            self.clf = pickle.load(model_path)
            print(f'Loaded pretrained model from {model_path}.')
