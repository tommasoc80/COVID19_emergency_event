from typing import List, Tuple, Callable, TypeVar, Any, overload, Dict
from typing import Optional as Maybe
from dataclasses import dataclass
from abc import ABC

from torch import Tensor, LongTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import numpy as np 

array = np.array 

Label = bool

T1 = TypeVar('T1')


@dataclass
class Sentence:
    no:     int
    text:   str


@dataclass
class AnnotatedSentence(Sentence):
    labels: [Label] * 8


class Model(ABC):

    def predict(self, tweets: List[Sentence]) -> List[int]:
        ...

    def predict_scores(self, tweets: List[Sentence]) -> array:
        ...
