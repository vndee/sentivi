from sentivi.base_model import DataLayer, ClassifierLayer


class DataLoader(DataLayer):
    def __init__(self):
        super(DataLoader, self).__init__()


class TextProcessor(DataLayer):
    def __init__(self):
        super(TextProcessor, self).__init__()


class Tokenizer(DataLayer):
    def __init__(self):
        super(Tokenizer, self).__init__()


class TextEncoder(DataLayer):
    def __init__(self):
        super(TextEncoder, self).__init__()


class NaiveBayesClassifier(ClassifierLayer):
    def __init__(self):
        super(NaiveBayesClassifier, self).__init__()


class SVMClassifier(ClassifierLayer):
    def __init__(self):
        super(SVMClassifier, self).__init__()


class CNNClassifier(ClassifierLayer):
    def __init__(self):
        super(CNNClassifier, self).__init__()


class RNNClassifier(ClassifierLayer):
    def __init__(self):
        super(RNNClassifier, self).__init__()


class TransformerClassifier(ClassifierLayer):
    def __init__(self):
        super(TransformerClassifier, self).__init__()


class MLPClassifier(ClassifierLayer):
    def __init__(self):
        super(MLPClassifier, self).__init__()


class DecisionTreeClassifier(ClassifierLayer):
    def __init__(self):
        super(DecisionTreeClassifier, self).__init__()
