import pandas as pd 
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import FeatureUnion

import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 

from typing import Tuple, Callable, List

array = np.array
Sentence = List[str]
WordEmbedder = Callable[[Sentence], array]
Sample = Tuple[array, array, array]
Samples = List[Sample]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sent: Sentence) -> array:
        # tokenize sentence and map to [sent_len X glove_dim] tensor
        sent_proc = _glove(sent)
        vectors = [word.vector for word in sent_proc]
        return array(vectors)
    return embedd


# frozen BERT pretrained embeddings with bert_dim=768
@torch.no_grad()
def bert_pretrained_embeddings() -> WordEmbedder:
    from transformers import BertTokenizer, BertModel 
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained('bert-base-uncased')
    def embedd(sent: Sentence) -> array:
        inps = tokenizer(sent, return_tensors="pt")
        vectors = bert(**inps).last_hidden_state.squeeze()
        return vectors.detach().numpy()
    return embedd


def get_word_embedder(embeddings: str) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See data_loading.py for valid embedding options')
        embedder = glove_embeddings(version)

    elif embeddings == 'bert':
        embedder = bert_pretrained_embeddings()

    elif embeddings == 'roberta':
        embedder = roberta_pretrained_embeddings()

    else:
        raise ValueError('See data_loading.py for valid embedding options')

    return embedder


def vectorize_label(label: str, lsa_components: int=100) -> array:
    # convert strings of ','-seperated bits to long tensor
    bits = list(map(int, label.split(',')))
    return array(bits, dtype=np.int32)


def vectorize_tf_idf(text: List[str], lsa_components: int=100, ngram_range_top: int=5
    ) -> Tuple[array, TruncatedSVD, TfidfVectorizer]:

    print(f'{lsa_components} LSA components.')
    # extract tf-idf features for each paragraph in word lvl
    tf_idf_vectorizer = TfidfVectorizer(ngram_range=(1, ngram_range_top), stop_words='english')

    tf_idf = tf_idf_vectorizer.fit_transform(text)

    # apply Latent Semantic Analysis (LSA) as truncated Singular Value Decomposition (SVD)
    # to reconstruct sparse tf-idf matrices in low-dimensions
    lsa = TruncatedSVD(n_components=lsa_components, random_state=42).fit(tf_idf)

    # return embeds but also transforms to use in test-time
    return lsa.transform(tf_idf), lsa, tf_idf_vectorizer


class EventClassesMultilabel(object):
    def __init__(self, csv_path: str, word_embedder: WordEmbedder) -> None:
        # load data from csv
        self.data = pd.read_csv(csv_path, sep=',', header=0).values.tolist()
        self.text, self.labels = zip(*self.data)
        
        # extract tf-idf vectors for input text
        # print('Extracting TF-IDF features - Latent Semantic Analysis...')
        # self.tf_idfs, self.lsa, self.tf_idf_vectorizer = vectorize_tf_idf(self.text)

        # convert input sentences to sequences of word-label vectors
        print('Extracting word embeddings...')
        self.text_embedds = list(map(word_embedder, self.text))
        self.label_embedds = list(map(vectorize_label, self.labels))
        
    def __getitem__(self, n: int) -> Sample:
        return (self.text_embedds[n], self.label_embedds[n])

    def __len__(self) -> int:
        return len(self.text) 

    def vectorize_tf_idf(self, *args, **kwargs):
        return vectorize_tf_idf(*args, **kwargs)


def vectorize_dataset(csv_path: str, embeddings: str) -> EventClassesMultilabel:
    return EventClassesMultilabel(csv_path, word_embedder=get_word_embedder(embeddings))


def collate_fn(pad_id: int) -> Callable[[Samples], Tuple[Tensor, LongTensor]]:
    def _collate_fn(batch: Samples) -> Tuple[Tensor, LongTensor]:
        xs, ys, vs = zip(*batch)
        xs = pad_sequence([Tensor(x) for x in xs], batch_first=True, padding_value=pad_id)
        ys = torch.stack([LongTensor(y) for y in ys], dim=0)
        vs = torch.stack([Tensor(v) for v in vs], dim=0)
        return (xs, ys, vs)
    return _collate_fn


def build_dataloaders(train_set: Samples, test_set:  Samples, bs: int=32, pad_id: int=-1) -> Tuple[DataLoader, DataLoader]:
    train_dl = DataLoader(train_set, batch_size=bs, shuffle=True, collate_fn=collate_fn(pad_id))
    test_dl = DataLoader(test_set, batch_size=bs, shuffle=False, collate_fn=collate_fn(pad_id))
    return train_dl, test_dl