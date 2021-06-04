from typing import List, Tuple, Callable, TypeVar, Any, overload, Dict, Set, Sequence, Generic
from typing import Optional as Maybe
from typing import Mapping as Map
from dataclasses import dataclass
from abc import ABC

from torch import Tensor, LongTensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch import long as longt 
from torch import float as floatt
from numpy import array

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


Tokenizer = Map[str, Sequence[int]]