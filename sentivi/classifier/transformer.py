from typing import Optional
from sentivi.base_model import ClassifierLayer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class TransformerClassifier(ClassifierLayer):
    TRANSFORMER_ALIASES = [
        'bert-base-multilingual-uncased',
        'bert-base-multilingual-cased',
        'xlm-roberta-base',
        'xlm-mlm-xnli15-1024',
        'xlm-mlm-tlm-xnli15-1024',
        'vinai/phobert-base',
        'vinai/phobert-large'
    ]

    def __init__(self,
                 num_labels: Optional[int] = 3,
                 language_model_shortcut: Optional[str] = 'vinai/phobert',
                 freeze_language_model: Optional[bool] = True,
                 batch_size: Optional[int] = 2,
                 warmup_steps: Optional[int] = 100,
                 weight_decay: Optional[float] = 0.01,
                 accumulation_steps: Optional[int] = 10,
                 save_steps: Optional[int] = 100,
                 device: Optional[str] = 'cpu',
                 *args,
                 **kwargs):
        super(TransformerClassifier, self).__init__()

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.accumulation_steps = accumulation_steps
        self.language_model_shortcut = language_model_shortcut
        self.save_steps = save_steps
        self.device = device

        assert language_model_shortcut in TransformerClassifier.TRANSFORMER_ALIASES, ValueError(
            f'language_model_shortcut must be in {TransformerClassifier.TRANSFORMER_ALIASES} '
            f'- not {language_model_shortcut}')

        self.clf_config = AutoConfig.from_pretrained(language_model_shortcut)
        self.clf_config.num_labels = num_labels

        self.tokenizer = AutoTokenizer.from_pretrained(language_model_shortcut)
        self.clf = AutoModelForSequenceClassification.from_pretrained(language_model_shortcut, config=self.clf_config)

        if freeze_language_model is True:
            for param in self.clf.base_model.parameters():
                param.requires_grad = True

    def __call__(self, data, *args, **kwargs):
        self.no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.clf.named_parameters() if not any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in self.clf.named_parameters() if any(nd in n for nd in self.no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=kwargs.get('learning_rate', 1e-5))
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=100, num_training_steps=1000)

        print(self.clf)

        # training

        # evaluate

    def predict(self, X, *args, **kwargs):
        pass

    def save(self, save_path, *args, **kwargs):
        pass

    def load(self, model_path):
        pass
