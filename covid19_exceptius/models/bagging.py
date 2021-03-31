from covid19_exceptius.types import *
from covid19_exceptius.utils.embeddings import WordEmbedder, make_word_embedder
from covid19_exceptius.utils.tf_idf import extract_tf_idfs, TfIdfTransform

from torch.nn import Module, Linear, GRU, Dropout, GELU, Sequential
import torch


# baseline model aggregating word-embedds and classifying with an MLP head
class BaselineModel(Module):
    def __init__(self, aggregator: Module, classifier: Module):
        super().__init__()
        self.aggregator = aggregator
        self.cls = classifier

    # optionally concat tf_idf reps to the pooled sent vectors
    def pool(self, x: Tensor, t: Maybe[Tensor]) -> Tensor:
        pooler = self.aggregator(x)
        if t is not None:
            pooler = torch.cat((pooler, t), dim=-1)
        return pooler

    def forward(self, inputs: Tuple[Tensor, Maybe[Tensor]]) -> Tensor:
        x, t = inputs
        features = self.pool(x, t)
        return self.cls(features)
        

# a wrapper for using a baseline model in test time
class BaselineModelTest(BaselineModel, Model):
    def __init__(self, aggregator: Module, classifier: Module, word_embedder: WordEmbedder, 
            device: str, tf_idf_transform: Maybe[TfIdfTransform] = None):
        super().__init__(aggregator, classifier)
        self.we = word_embedder 
        self.tf_idf = tf_idf_transform
        self.device = device

    def embedd(self, sents: List[Sentence]) -> Tuple[Tensor, Maybe[Tensor]]:
        text = [sent.text for sent in sents]
        word_embedds = torch.tensor(self.we(text), dtype=torch.float, device=self.device)
        tf_idfs = None
        if self.tf_idf is not None:
            tf_idfs = torch.tensor(self.tf_idf.transform(text), dtype=torch.float, device=self.device)
        return word_embedds, tf_idfs

    def predict(self, sents: List[Sentence]) -> List[int]:
        inputs = self.embedd(sents)
        return self.forward(inputs).sigmoid().round().long().cpu().tolist()

    def predict_scores(self, sents: List[Sentence]) -> array:
        inputs = self.embedd(sents)
        return self.forward(inputs).sigmoid().cpu().tolist()
        

# bag of embedds aggregation with average pooling
class BagOfEmbeddings(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(dim=1).squeeze() # B, S, D -> B, D


# bi-GRU contextualization as aggregator
class RNNContext(Module):
    def __init__(self, inp_dim: int, hidden_dim: int):
        super().__init__()
        self.rnn = GRU(input_size=inp_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

    def forward(self, x: Tensor) -> Tensor:
        dh = self.rnn.hidden_size
        h, _ = self.rnn(x)
        context = torch.cat((h[:, -1, :dh], h[:, 0, dh:]), dim=-1)
        return context 


# 2-layer MLP head for classification
class MLPHead(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.hidden = Linear(in_features=inp_dim, out_features=hidden_dim)
        self.out = Linear(in_features=hidden_dim, out_features=num_classes)
        self.gelu = GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.out(x)



models_dict = {
    'BoE'  : MultiLabelBagOfEmbeddings,
    'MLP'  : MultiLabelMLP,
    'LSTM' : MultiLabelLSTM,
    'CNN'  : MultiLabelCNN,
    'TRM'  : MultiLabelTransformer,
    'TRM+c': MultiLabelTransformerWithContextLast,
    'c+TRM': MultiLabelTransformerWithContextFirst
}


def setup_model_kwargs(model_idx: str, kwargs: Dict[str, float]) -> Dict[str, float]:
    if model_idx not in list(models_dict.keys()):
        print('Please enter a valid model key:')
        print('(BoE) - bag of embeddings')
        print('(MLP) - multi-layer perceptron')
        print('(LSTM) - Bi-LSTM contextualization')
        print('(CNN) - 1d CNN features')
        print('(TRM) - Transformer Encoder')
        print('(TRM+c)' - 'Transformer Encoder + Bi-LSTM contextualization')
        print('(c+TRM)' - 'Bi-LSTM contextualization + Transformer Encoder')
        return

    elif model_idx == 'BoE':
        return {'inp_dim': kwargs['embedd_dim'], 'num_classes': kwargs['num_classes'], 
                'with_tf_idf': kwargs['with_tf_idf']}

    elif model_idx == 'MLP':
        return {'inp_dim': kwargs['embedd_dim'], 'inter_dim': kwargs['inter_dim'],
                'num_classes': kwargs['num_classes'], 'dropout': kwargs['dropout'], 
                'with_tf_idf': kwargs['with_tf_idf']}

    elif model_idx == 'LSTM':
        return {'inp_dim': kwargs['embedd_dim'], 'hidden_dim': kwargs['model_dim'],
                'inter_dim': kwargs['inter_dim'], 'num_layers': kwargs['num_layers'], 
                'num_classes': kwargs['num_classes'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}

    elif model_idx == 'CNN':
        return {'hidden_dim': kwargs['model_dim'], 'num_classes': kwargs['num_classes'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}

    else: # TRM modules
        return {'inp_dim': kwargs['embedd_dim'], 'model_dim': kwargs['model_dim'],
                'num_layers': kwargs['num_layers'], 'num_classes': kwargs['num_classes'],
                'num_heads': kwargs['num_heads'], 'pwff_dim': 2*kwargs['model_dim'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}