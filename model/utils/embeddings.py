from covid19_exceptius.types import *
import torch

WordEmbedder = Callable[[Sentence], array]


# pre-trained GloVe embeddings with glove_dim=300
def glove_embeddings(version: str) -> WordEmbedder:
    import spacy
    _glove = spacy.load(f'en_core_web_{version}') 
    def embedd(sent: Sentence) -> array:
        sent_proc = _glove(sent)
        return array([word.vector for word in sent_proc])
    def embedd_many(sents: List[Sentence]) -> List[array]:
        return list(map(embedd, sents))
    return embedd_many


# last hidden layer representations of a pre-trained BERT encoder
@torch.no_grad()
def frozen_bert_embeddings(name: str, **kwargs) -> WordEmbedder:
    from covid19_exceptius.models.bert import make_classification_model, pad_sequence
    model = make_classification_model(name=name, **kwargs)
    def embedd(sents: List[Sentence]) -> List[array]:
        pad_id = model.tokenizer.pad_token_id
        tokens = pad_sequence(model.tensorize_unlabeled(sents), padding_value=pad_id)
        attention_mask = tokens.ne(pad_id)
        hidden, _ = model.core.forward(tokens, attention_mask, output_hidden_states=False, return_dict=False)
        return list(hidden.cpu().numpy())
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


