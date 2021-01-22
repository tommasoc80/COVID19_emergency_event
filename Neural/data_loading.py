import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split

#DL utils
import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

from typing import Tuple, Callable, List

Sentence = List[str]
WordEmbedder = Callable[[Sentence], Tensor]
Sample = Tuple[Tensor, LongTensor]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings():
    import spacy
    _glove = spacy.load('en_core_web_md') 
    def embedd(sent: Sentence) -> Tensor:
        # tokenize sentence and map to [sent_len X glove_dim] tensor
        sent_proc = _glove(sent)
        vectors = [word.vector for word in sent_proc]
        return torch.tensor(vectors)

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

    def random_train_test_split(self) -> Tuple[List[Sample], List[Sample]]:
        X_train, X_test, Y_train, Y_test = train_test_split(self.text_embedds, \
            self.label_embedds, test_size = 0.20, random_state = 42)
        return list(zip(X_train, Y_train)), list(zip(X_test, Y_test))

    def __getitem__(self, n: int) -> Sample:
        return (self.text_embedds[n], self.label_embedds[n])

    def __len__(self) -> int:
        return len(self.text)


def collate_fn(batch: List[Sample]) -> Tuple[Tensor, LongTensor]:
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True), torch.stack(ys, dim=0)
 

def get_datasets(csv_path: str) -> Tuple[List[Sample], List[Sample]]:
    dataset = EventClassesMultilabel(csv_path, word_embedder=glove_embeddings())
    train_set, test_set = dataset.random_train_test_split()
    return train_set, test_set


def build_dataloaders(train_set: List[Sample], test_set:  List[Sample], bs: int=32) -> Tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn)
    test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, collate_fn=collate_fn)
    return train_dl, test_dl  