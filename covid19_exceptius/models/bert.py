from covid19_exceptius.types import *
from covid19_exceptius.preprocessing import read_labeled, read_unlabeled
from covid19_exceptius.utils.masker import get_masker

from torch import tensor, stack, load
from torch.nn import Module, Linear, Dropout, Sequential, LayerNorm
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class BertoidLM(Module):
    def __init__(self, name: str, max_length: Maybe[int] = None, token_name: Maybe[str] = None, model_dim: int = 768):
        super().__init__()
        self.token_name = name if token_name is None else token_name
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.token_name, use_fast=False)
        self.max_length = max_length
        self.head = Sequential(Linear(model_dim, model_dim),
                               LayerNorm(model_dim, eps=1e-05),
                               Linear(model_dim, self.tokenizer.vocab_size)
                              )
        self.masker = get_masker(self.tokenizer)

    def tokenize_and_mask(self, sents: List[Sentence]) -> List[Tuple[Tensor, ...]]:
        def to_longt(seq: List[int]) -> Tensor:
            return [tensor(s, dtype=longt) for s in seq]
        
        sents = [self.tokenizer.encode(s.text, truncation=True, max_length=self.max_length) for s in sents]
        mask_ids, masked_sents = zip(*list(map(self.masker, sents)))
        return to_longt(masked_sents), to_longt(sents), to_longt(mask_ids)

    def forward(self, x: Tensor, mlm_mask: Tensor) -> Tensor:
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        hidden, _ = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.head(hidden[mlm_mask == 1])
        

class BertoidSentClassification(Module, Model):
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

    def tensorize_labeled(self, sents: List[AnnotatedSentence]) -> List[Tuple[Tensor, Tensor]]:
        return [tokenize_labeled(sent, self.tokenizer, max_length=self.max_length) for sent in sents]

    def tensorize_unlabeled(self, sents: List[Sentence]) -> List[Tensor]:
        return [tokenize_unlabeled(sent, self.tokenizer, max_length=self.max_length) for sent in sents]

    def forward(self, x: Tensor) -> Tensor:
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        _, cls = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.classifier(self.dropout(cls))

    def predict_scores(self, sents: List[Sentence], device: str) -> List[List[float]]:
        tensorized = pad_sequence(self.tensorize_unlabeled(sents), padding_value=self.tokenizer.pad_token_id).to(device)
        return self.forward(tensorized).sigmoid().cpu().tolist()

    def predict(self, sents: List[Sentence], device: str) -> List[List[int]]:
        tensorized = pad_sequence(self.tensorize_unlabeled(sents), padding_value=self.tokenizer.pad_token_id).to(device)
        return self.forward(tensorized).sigmoid().round().long().cpu().tolist()


def collate_tuples(pairs: List[Tuple[Tensor, ...]], padding_values: Tuple[int], device: str) -> Tuple[Tensor, ...]:
    zipped = zip(*pairs)
    return tuple([pad_sequence(tens, pad).to(device) for tens, pad in zip(zipped, padding_values)])


def pad_sequence(xs: List[Tensor], padding_value: int) -> Tensor:
    return _pad_sequence(xs, batch_first=True, padding_value=padding_value)


def tokenize_text(text: str, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tensor(tokenizer.encode(text, truncation=True, **kwargs), dtype=longt)


def tokenize_labels(labels: List[Label]) -> Tensor:
    return tensor([0 if label is False or label is None else 1 for label in labels], dtype=longt)


def tokenize_unlabeled(sent: Sentence, tokenizer: AutoTokenizer, **kwargs) -> Tensor:
    return tokenize_text(sent.text, tokenizer, **kwargs)


def tokenize_labeled(sent: AnnotatedSentence, tokenizer: AutoTokenizer, **kwargs) -> Tuple[Tensor, Tensor]:
    return tokenize_unlabeled(sent, tokenizer, **kwargs), tokenize_labels(sent.labels)


def make_labeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tuple[Tensor, Tensor]]:
    return [tokenize_labeled(sent, tokenizer, **kwargs) for sent in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tensor]:
    return [tokenize_unlabeled(sent, tokenizer, **kwargs) for sent in read_unlabeled(path)]


def make_model(name: str, **kwargs) -> BertoidSentClassification:
    # todo: find all applicable models
    if name == 'eng-bert':
        return BertoidSentClassification(name='bert-base-uncased', **kwargs)
    elif name == 'eng-legal':
        return BertoidSentClassification(name='nlpaueb/legal-bert-base-uncased', **kwargs)

    # multi-lingual models
    elif name == 'mbert':
        return BertoidSentClassification(name='bert-base-multilingual-cased', **kwargs)
    elif name == 'xlm':
        return BertoidSentClassification(name='xlm-roberta-base', **kwargs)
    elif name == 'mbert-xnli':
        return BertoidSentClassification(name='joeddav/xlm-roberta-large-xnli', model_dim=1024, **kwargs)
    elif name == 'mbert-microsoft':
        return BertoidSentClassification(name='microsoft/Multilingual-MiniLM-L12-H384', model_dim=384, token_name='xlm-roberta-base', **kwargs)
    elif name == 'mbert-sentiment':
        return BertoidSentClassification(name='socialmediaie/TRAC2020_ALL_C_bert-base-multilingual-uncased', **kwargs)
    elif name == 'mbert-toxic':
        return BertoidSentClassification(name='unitary/multilingual-toxic-xlm-roberta', **kwargs)
    else:
        raise ValueError(f'unknown name {name}')


def annotate_files(files: List[str], model: BertoidSentClassification, save_path: str, device: str = 'cuda'):
    import os
    from tqdm import tqdm
    from math import ceil
    model.eval().to(device)

    for file in tqdm(files):
        filename = file.split('/')[-1].split('.')[0]
        with open(file, 'r') as f:
            lines = f.readlines()
        text = [Sentence(no='', text=l.split('\t')[0]) for l in lines]
        bs = 16; num_batches = ceil(len(text) / bs)
        predictions = []
        for batch_idx in range(num_batches):
            _text = text[batch_idx * bs : (1 + batch_idx) * bs]
            predictions.extend(model.predict(_text, device))
        to_write = ['\t'.join([line.text, *list(map(str, preds))]) for line, preds in zip(text, predictions)]
        with open(os.path.join(save_path, filename + '.tsv'), 'w') as f:
            f.write('\n'.join(to_write))