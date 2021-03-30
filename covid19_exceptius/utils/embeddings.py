from covid19_exceptius.types import *
from covid19_exceptius.bertoids.bert import Bertoid
import torch

WordEmbedder = Callable[[Sentence], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sents: Sentences) -> array:
        sents_proc = [_glove(sent) for sent in sents]
        return array([word.vector for word in sent] for sent in sents_proc)
    return embedd


# last hidden layer representations of a pre-trained BERT encoder
@torch.no_grad()
def frozen_bert_embeddings(name: str, **kwargs) -> WordEmbedder:
    model = Bertoid(name=name, **kwargs)
    def embedd(sents: Sentences) -> array:
        tokens = stack(model.tensorize_labeled(sents))
        attention_mask = tokens.ne(tokenizer.pad_token_id)
        hidden, _ = model(tokens, attention_mask, output_hidden_states=True, return_dict=False).squeeze()
        return hidden.cpu().numpy()
    return embedd 


# make word embedder function
def make_word_embedder(embeddings: str) -> WordEmbedder:
    if embeddings.startswith('glove_'):
        version = embeddings.split('_')[1]
        if version not in ['md', 'lg']:
            raise ValueError('See utils/embeddings.py for valid embedding options')
        embedder = glove_embeddings(version)

    elif 'bert' in embeddings:
        embedder = frozen_bert_embeddings(name=embeddings)


    else:
        raise ValueError('See utils/embeddings.py for valid embedding options')

    return embedder


