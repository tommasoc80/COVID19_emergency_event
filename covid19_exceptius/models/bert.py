from covid19_exceptius.types import *
from covid19_exceptius.utils.masker import get_masker, Masker

from torch import tensor, stack, load, no_grad
from torch.nn import Module, Linear, Dropout, Sequential, LayerNorm
from torch.nn.utils.rnn import pad_sequence as _pad_sequence

from transformers import AutoModel, AutoTokenizer


class BertoidLM(Module):
    def __init__(self, name: str, max_length: Maybe[int] = None, token_name: Maybe[str] = None, model_dim: int = 768):
        super().__init__()
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name if token_name is None else token_name, use_fast=False)
        self.max_length = max_length
        self.head = Sequential(Linear(model_dim, model_dim),
                               LayerNorm(model_dim, eps=1e-05),
                               Linear(model_dim, self.tokenizer.vocab_size)
                              )
        self.masker = get_masker(self.tokenizer)

    def tokenize_and_mask(self, sents: List[Sentence]) -> List[Tuple[Tensor, ...]]:
        sents = [self.tokenizer.encode(s.text, truncation=True, max_length=self.max_length) for s in sents]
        return self.mask(sents)
        
    def mask(self, tokens: List[Sequence[int]]) -> List[Tuple[Tensor, ...]]:
        def to_longt(seq: Sequence[int]) -> List[Tensor]:
            return [tensor(s, dtype=longt) for s in seq]
        
        mask_ids, masked_tokens = zip(*list(map(self.masker, tokens)))
        return to_longt(masked_tokens), to_longt(tokens), to_longt(mask_ids)

    def forward(self, x: Tensor, mlm_mask: Tensor) -> Tensor:
        attention_mask = x.ne(self.tokenizer.pad_token_id)
        hidden, _ = self.core(x, attention_mask, output_hidden_states=False, return_dict=False)
        return self.head(hidden[mlm_mask == 1])

    def load_core(self, load_path: str):
        self.core.load_state_dict(load(load_path))
        

class BertoidSentClassification(Module, Model):
    def __init__(self, name: str, model_dim: int = 768, num_classes: int = 8,
                 dropout_rate: float = 0.5, max_length: Maybe[int] = None, 
                 token_name: Maybe[str] = None):
        super().__init__()
        self.core = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name if token_name is None else token_name, use_fast=False)
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

    @no_grad()
    def predict_scores(self, sents: List[Sentence], device: str) -> List[List[float]]:
        self.eval()
        tensorized = pad_sequence(self.tensorize_unlabeled(sents), padding_value=self.tokenizer.pad_token_id).to(device)
        return self.forward(tensorized).sigmoid().cpu().tolist()

    @no_grad()
    def predict(self, sents: List[Sentence], device: str) -> List[List[int]]:
        self.eval()
        tensorized = pad_sequence(self.tensorize_unlabeled(sents), padding_value=self.tokenizer.pad_token_id).to(device)
        return self.forward(tensorized).sigmoid().round().long().cpu().tolist()
    
    def load_core(self, load_path: str):
        self.core.load_state_dict(load(load_path))


def collate_with_mask(tokens: List[Sequence[int]], mask_fn: Masker, padding_value: int, device: str) -> Tuple[Tensor, ...]:
    masked_tokens, tokens, mask_ids = mask_fn(tokens)
    return (pad_sequence(masked_tokens, padding_value).to(device), 
            pad_sequence(tokens, padding_value).to(device), 
            pad_sequence(mask_ids, -1).to(device))
    

def collate_tuples(pairs: List[Tuple[Tensor, ...]], padding_value: int, device: str) -> Tuple[Tensor, ...]:
    zipped = zip(*pairs)
    return tuple([pad_sequence(x, padding_value).to(device) for x in zipped])


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
    from covid19_exceptius.preprocessing import read_labeled
    return [tokenize_labeled(sent, tokenizer, **kwargs) for sent in read_labeled(path)]


def make_unlabeled_dataset(path: str, tokenizer: AutoTokenizer, **kwargs) -> List[Tensor]:
    from covid19_exceptius.preprocessing import read_unlabeled
    return [tokenize_unlabeled(sent, tokenizer, **kwargs) for sent in read_unlabeled(path)]


def make_pretrain_dataset(path: str, tokenizer: AutoTokenizer, with_save: Maybe[str] = None, **kwargs) -> List[Tensor]:
    from covid19_exceptius.preprocessing import prepare_pretrain_corpus
    return prepare_pretrain_corpus(path, tokenizer, save_path=with_save)


def make_model(name: str, version: str, **kwargs):
    if version == 'mlm':
        model = BertoidLM
    elif version == 'classifier':
        model = BertoidSentClassification
    else:
        raise ValueError(f'unknown model verison {version}')

    if name == 'eng-bert':
        return model(name='bert-base-uncased', **kwargs)
    elif name == 'eng-legal':
        return model(name='nlpaueb/legal-bert-base-uncased', **kwargs)

    # multi-lingual models
    elif name == 'mbert':
        return model(name='bert-base-multilingual-cased', **kwargs)
    elif name == 'xlm':
        return model(name='xlm-roberta-base', **kwargs)
    elif name == 'mbert-xnli':
        return model(name='joeddav/xlm-roberta-large-xnli', model_dim=1024, **kwargs)
    elif name == 'mbert-microsoft':
        return model(name='microsoft/Multilingual-MiniLM-L12-H384', model_dim=384, token_name='xlm-roberta-base', **kwargs)
    else:
        raise ValueError(f'unknown name {name}')    


def annotate_files(files: List[str], model: BertoidSentClassification, save_path: str, device: str = 'cuda'):
    from covid19_exceptius.preprocessing import read_processed
    from tqdm import tqdm
    from math import ceil
    import os

    model = model.eval().to(device)

    for file in tqdm(files):
        filename = file.split('/')[-1].split('.')[0]
        text = read_processed(file)
        bs = 16; num_batches = ceil(len(text) / bs)
        predictions = []
        for batch_idx in range(num_batches):
            _text = text[batch_idx * bs : (1 + batch_idx) * bs]
            predictions.extend(model.predict(_text, device))
        to_write = ['\t'.join([line.text, *list(map(str, preds))]) for line, preds in zip(text, predictions)]
        with open(os.path.join(save_path, filename + '.tsv'), 'w') as f:
            f.write('\n'.join(to_write))