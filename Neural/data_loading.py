import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

from typing import Tuple, Callable, List

Sentence = List[str]
WordEmbedder = Callable[[Sentence], Tensor]
Sample = Tuple[Tensor, LongTensor]
Samples = List[Sample]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings() -> WordEmbedder:
    import spacy
    _glove = spacy.load('en_core_web_md') 
    def embedd(sent: Sentence) -> Tensor:
        # tokenize sentence and map to [sent_len X glove_dim] tensor
        sent_proc = _glove(sent)
        vectors = [word.vector for word in sent_proc]
        return torch.tensor(vectors)
    return embedd


# frozen BERT pretrained embeddings with bert_dim=768
@torch.no_grad()
def bert_pretrained_embeddings() -> WordEmbedder:
    from transformers import BertTokenizer, BertModel 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    def embedd(sent: Sentence) -> Tensor:
        inps = tokenizer(sent, return_tensors="pt")
        vectors = bert(**inps).last_hidden_state.squeeze()
        return vectors
    return embedd


def vectorize_label(label: str) -> LongTensor:
    # convert strings of ','-seperated bits to long tensor
    bits = list(map(int, label.split(',')))
    return torch.tensor(bits)


class EventClassesMultilabel(object):
    def __init__(self, csv_path: str, word_embedder: WordEmbedder) -> None:
        self.data = pd.read_csv(csv_path, sep=',', header=0).values.tolist()
        self.text, self.labels = zip(*self.data)
        self.text_embedds = list(map(word_embedder, self.text))
        self.label_embedds = list(map(vectorize_label, self.labels))

    def random_train_test_split(self) -> Tuple[Samples, Samples]:
        X_train, X_test, Y_train, Y_test = train_test_split(self.text_embedds, \
            self.label_embedds, test_size = 0.20, random_state = 42)
        return list(zip(X_train, Y_train)), list(zip(X_test, Y_test))

    def __getitem__(self, n: int) -> Sample:
        return (self.text_embedds[n], self.label_embedds[n])

    def __len__(self) -> int:
        return len(self.text)


def vectorize_dataset(csv_path: str, embeddings: str) -> EventClassesMultilabel:
    if embeddings == 'glove':
        embedder = glove_embeddings()
    elif embeddings == 'bert':
        embedder = bert_pretrained_embeddings()
    else:
        raise ValueError('See data_loading.py for valid embedding options')

    return EventClassesMultilabel(csv_path, word_embedder=embedder)


def collate_fn(pad_id: int) -> Callable[[Samples], Tuple[Tensor, LongTensor]]:
    def _collate_fn(batch: Samples) -> Tuple[Tensor, LongTensor]:
        xs, ys = zip(*batch)
        return pad_sequence(xs, batch_first=True, padding_value=pad_id), torch.stack(ys, dim=0)
    return _collate_fn


def build_dataloaders(train_set: Samples, test_set:  Samples, bs: int=32, pad_id: int=-1) -> Tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn(pad_id))
    test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, collate_fn=collate_fn(pad_id))
    return train_dl, test_dl