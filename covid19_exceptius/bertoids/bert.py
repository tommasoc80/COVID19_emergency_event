from covid19_exceptius.types import *
from covid19_exceptius.utils.preprocessing import read_labeled, read_unlabeled

from torch import tensor, long, stack
from torch.nn import Module, Linear, Dropout
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class Bertoid(Module, Model):
    def __init__(self, name: str, model_dim: int = 768, num_classes: int = 8,
            dropout_rate: float = 0.5, max_length: Maybe[int] = None, 
            token_name: Maybe[str] = None):
        super().__init__()
        self.token_name = name if token_name is None else token_name
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name, use_fast=False)
        self.max_length = max_length
        self.dropout = Dropout(dropout_rate)
        self.classifier = Linear(model_dim, num_classes)

    def tensorize_labeled(self, tweets: List[AnnotatedSentence]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(tweet, self.tokenizer, max_length=self.max_length) for tweet in tweets]

    def tensorize_unlabeled(self, tweets: List[Sentence]) -> List[Tensor]:
        return [tokenize_unlabeled(tweet, self.tokenizer, max_length=self.max_length) for tweet in tweets]

    def forward(self, x: Tensor):
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict_scores(self, tweets: List[Sentence]) -> List[List[float]]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        return self.forward(tensorized).sigmoid().cpu().tolist()

    def predict(self, tweets: List[Sentence]) -> List[str]:
        tensorized = pad_sequence(self.tensorize_unlabeled(tweets), padding_value=self.tokenizer.pad_token_id)
        preds = self.forward(tensorized).sigmoid().round().long().cpu().tolist()
        return [preds_to_str(sample) for sample in preds]


def collate_tuples(pairs: List[Tuple[Tensor, Tensor]], padding_value: int) -> Tuple[Tensor, Tensor]:
    xs, ys = list(zip(*pairs))
    return pad_sequence(xs, padding_value), stack(ys)


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def longt(x: Any) -> Tensor:
    return tensor(x, dtype=long)


def tokenize_text(text: str, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return longt(tokenizer.encode(text, truncation=True, **kwargs))


def tokenize_labels(labels: List[Label]) -> Tensor:
    return longt([0 if label is False or label is None else 1 for label in labels])


def tokenize_unlabeled(tweet: Sentence, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tokenize_text(tweet.text, tokenizer, **kwargs)


def tokenize_labeled(tweet: AnnotatedSentence, tokenizer: AutoTokenizer, **kwargs) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(tweet, tokenizer, **kwargs), tokenize_labels(tweet.labels)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(tweet, tokenizer, **kwargs) for tweet in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tensor]:
    return [tokenize_unlabeled(tweet, tokenizer, **kwargs) for tweet in read_unlabeled(path)]


def make_model(name: str) -> Bertoid:
    # todo: find all applicable models
    if name == 'del-covid':
        return Bertoid(name='digitalepidemiologylab/covid-twitter-bert-v2', model_dim=1024)
    elif name == 'vinai-covid':
        return Bertoid(name='vinai/bertweet-covid19-base-cased', max_length=128)
    
    # multi-lingual models
    elif name == 'multi-bert':
        return Bertoid(name='bert-base-multilingual-cased')
    elif name == 'multi-xlm':
        return Bertoid(name='xlm-roberta-base')
    elif name == 'multi-xnli':
        return Bertoid(name='joeddav/xlm-roberta-large-xnli', model_dim=1024)
    elif name == 'multi-microsoft':
        return Bertoid(name='microsoft/Multilingual-MiniLM-L12-H384', model_dim=384, token_name='xlm-roberta-base')
    elif name == 'multi-sentiment':
        return Bertoid(name='socialmediaie/TRAC2020_ALL_C_bert-base-multilingual-uncased')
    elif name == 'multi-toxic':
        return Bertoid(name='unitary/multilingual-toxic-xlm-roberta')
    else:
        raise ValueError(f'unknown name {name}')