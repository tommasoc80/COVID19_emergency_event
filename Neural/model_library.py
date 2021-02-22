import torch
from torch import Tensor, LongTensor
from torch.nn import (Module, Linear, Conv1d, LSTM, Dropout, MaxPool1d,
                    Sequential, GELU, AdaptiveAvgPool1d)
import torch.nn.functional as F 

from transformer import positional_encoding, TransformerEncoderLayer

from typing import Dict


# average pooling -> N-to-8 logits for binary cross-entropy
class MultiLabelBagOfEmbeddings(Module):
    def __init__(self, inp_dim: int, num_classes: int, with_tf_idf: int=0, pad_id: int=-1):
        super().__init__()
        self.pad_id = pad_id
        self.forward = self._forward if not with_tf_idf else self._forward_tfIdf
        self.ff = Linear(in_features=inp_dim+with_tf_idf, out_features=num_classes)

    def _forward(self, x: Tensor, t: Tensor) -> Tensor:
        x[x==self.pad_id] = 0 # zero out padded values 
        pooler = x.mean(dim=1)
        out = self.ff(pooler)
        return out 

    def _forward_tfIdf(self, x: Tensor, t: Tensor) -> Tensor:
        x[x==self.pad_id] = 0 # zero out padded values 
        pooler = torch.cat((x.mean(dim=1), t), dim=-1)
        out = self.ff(pooler)
        return out 

# average pooling -> hidden layer -> N-to-8 logits for binary cross-entropy
class MultiLabelMLP(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_classes: int, dropout: float, with_tf_idf: int=0, pad_id: int=-1):
        super().__init__()
        self.pad_id = pad_id
        self.dropout = Dropout(p=dropout)
        self.forward = self._forward if not with_tf_idf else self._forward_tfIdf
        self.hidden = Linear(in_features=inp_dim+with_tf_idf, out_features=hidden_dim)
        self.out = Linear(in_features=hidden_dim, out_features=num_classes)

    def _forward(self, x: Tensor, t: Tensor) -> Tensor:
        x[x==self.pad_id] = 0 # zero out padded values 
        pooler = x.mean(dim=1)
        h = self.hidden(pooler)
        h = F.gelu(h)
        h = self.dropout(h)
        out = self.out(h)
        return out 

    def _forward_tfIdf(self, x: Tensor, t: Tensor) -> Tensor:
        x[x==self.pad_id] = 0 # zero out padded values 
        pooler = torch.cat((x.mean(dim=1), t), dim=-1)
        h = self.hidden(pooler)
        h = torch.tanh(h)
        h = self.dropout(h)
        out = self.out(h)
        return out 


# under construction
class MultiLabelCNN(Module):
    def __init__(self, num_blocks: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = Dropout(p=dropout)
        self.pooling_dim = 1024 // 2**(3+num_blocks)
        self.cnn1 = self.conv_block(in_channels=1, out_channels=16, conv_kernel=7, pool_kernel=3, conv_stride=4)
        self.cnns = Sequential(self.cnn1, *[self.conv_block(2**(3+block), 2**(4+block), 7, 3, 4) for block in range(1,num_blocks)])
        self.avg_pool = AdaptiveAvgPool1d(self.pooling_dim)
        self.ff = Linear(in_features=1024, out_features=num_classes)
        self.forward = self._forward if not with_tf_idf else self._forward_tfIdf

    def conv_block(self, in_channels: int, out_channels:int, conv_kernel:int, pool_kernel:int, conv_stride:int):
        return Sequential(*[
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=conv_kernel, stride=conv_stride),
            GELU(),
            MaxPool1d(kernel_size=pool_kernel)
            ])

    def _forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embedd_dim = x.shape 
        x = x.view(batch_size, 1, seq_len * embedd_dim)
        feats = self.cnns(x)
        pooler = self.avg_pool(feats).flatten(-2)
        pooler = self.dropout(pooler)
        out = self.ff(pooler)
        return out

    def _forward_tfIdf(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embedd_dim = x.shape 
        x = x.view(batch_size, 1, seq_len * embedd_dim)
        feats = self.cnns(x)
        pooler = self.avg_pool(feats).flatten(-2)
        pooler = self.dropout(pooler)
        out = self.ff(pooler)
        return out


# Bi-LSTM layers -> context vector -> N-to-8 logits for binary cross-entropy
class MultiLabelLSTM(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float, 
                 with_tf_idf: int=0):
        super().__init__()
        self.num_layers = num_layers
        self.forward = self._forward if not with_tf_idf else self._forward_tfIdf
        self.lstm = LSTM(input_size=inp_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.ff = Linear(in_features=2*hidden_dim + with_tf_idf, out_features=num_classes)
        self.dropout = Dropout(p=dropout)

    def _forward(self, x: Tensor, t: Tensor) -> Tensor:
        hiddens, _ = self.lstm(x)
        context = torch.cat((hiddens[:,-1,:self.lstm.hidden_size], hiddens[:,0,self.lstm.hidden_size:]), dim=-1)
        context = self.dropout(context)
        out = self.ff(context) 
        return out

    def _forward_tfIdf(self, x: Tensor, t: Tensor) -> Tensor:
        hiddens, _ = self.lstm(x)
        context = torch.cat((hiddens[:,-1,:self.lstm.hidden_size], hiddens[:,0,self.lstm.hidden_size:]), dim=-1)
        context = self.dropout(torch.cat((context, torch.tanh(t)), dim=-1))
        out = self.ff(context) 
        return out


# Transformer Encoder -> average pooling -> N-to-8 logits for binary cross-entropy
class MultiLabelTransformer(Module):
    def __init__(self, inp_dim: int, model_dim: int, num_heads: int, num_layers: int, num_classes: int, pwff_dim: int, 
        dropout: float, pad_id: int=-1, with_tf_idf: int=0):
        super().__init__()
        self.forward = self._forward if not with_tf_idf else self._forward_tfIdf
        self.encoder = Sequential(*[TransformerEncoderLayer(model_dim, inp_dim, num_heads, pwff_dim, dropout) for _ in range(num_layers)])
        self.ff = Linear(in_features=model_dim + with_tf_idf, out_features=num_classes)
        self.dropout = Dropout(p=dropout)
        self.pad_id = pad_id

    def _forward(self, x: Tensor, t: Tensor) -> Tensor:
        # we assume word ids are already embedded in vectors
        batch_size, seq_len, embedd_dim = x.shape
        padding_mask = (x!=self.pad_id).sum(dim=-1).bool().unsqueeze(1).repeat(1, seq_len, 1).long().to(x.device)
        x = x + positional_encoding(batch_size, seq_len, embedd_dim, device=x.device)
        pooler = self.encoder((x, padding_mask))[0].mean(dim=1)
        pooler = self.dropout(pooler)
        out = self.ff(pooler)
        return out

    def _forward_tfIdf(self, x: Tensor, t: Tensor) -> Tensor:
        # we assume word ids are already embedded in vectors
        batch_size, seq_len, embedd_dim = x.shape
        padding_mask = (x!=self.pad_id).sum(dim=-1).bool().unsqueeze(1).repeat(1, seq_len, 1).long().to(x.device)
        x = x + positional_encoding(batch_size, seq_len, embedd_dim, device=x.device)
        pooler = self.encoder((x, padding_mask))[0].mean(dim=1)
        pooler = self.dropout(torch.cat((pooler, v), dim=-1))
        out = self.ff(pooler)
        return out


# Transformer Encoder + Bi-LSTM contextualization -> N-to-8 logits for binary cross-entropy
class MultiLabelTransformerWithContextLast(Module):
    def __init__(self, inp_dim: int, model_dim: int, num_heads: int, num_layers: int, num_classes: int, pwff_dim: int, 
        dropout: float, pad_id: int=-1):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Sequential(*[TransformerEncoderLayer(model_dim, inp_dim, num_heads, pwff_dim, dropout) for _ in range(num_layers)])
        self.lstm = LSTM(input_size=model_dim, hidden_size=model_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.ff = Linear(in_features=model_dim, out_features=num_classes)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embedd_dim = x.shape
        padding_mask = (x!=self.pad_id).sum(dim=-1).bool().unsqueeze(1).repeat(1, seq_len, 1).long().to(x.device)
        x = x + positional_encoding(batch_size, seq_len, embedd_dim, device=x.device)
        x = self.encoder((x,padding_mask))[0]
        x = self.dropout(x)
        x, _ = self.lstm(x)
        context = torch.cat((x[:,-1,:self.lstm.hidden_size], x[:,0,self.lstm.hidden_size:]), dim=-1)
        context = self.dropout(context)
        out = self.ff(context)
        return out


# Bi-LSTM contextualization + Transformer Encoder -> N-to-8 logits for binary cross-entropy
class MultiLabelTransformerWithContextFirst(Module):
    def __init__(self, inp_dim: int, model_dim: int, num_heads: int, num_layers: int, num_classes: int, pwff_dim: int, 
        dropout: float, pad_id: int=-1):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = Sequential(*[TransformerEncoderLayer(model_dim, inp_dim, num_heads, pwff_dim, dropout) for _ in range(num_layers)])
        self.lstm = LSTM(input_size=inp_dim, hidden_size=inp_dim//2, num_layers=1, batch_first=True, bidirectional=True)
        self.ff = Linear(in_features=inp_dim, out_features=num_classes)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, seq_len, embedd_dim = x.shape
        padding_mask = (x!=self.pad_id).sum(dim=-1).bool().unsqueeze(1).repeat(1, seq_len, 1).long().to(x.device)
        x, _ = self.lstm(x)
        x = x + positional_encoding(batch_size, seq_len, embedd_dim, device=x.device)
        x = self.encoder((x,padding_mask))[0]
        context = torch.cat((x[:,-1,:self.lstm.hidden_size], x[:,0,self.lstm.hidden_size:]), dim=-1)
        context = self.dropout(context)
        out = self.ff(context)
        return out


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
        return {'inp_dim': kwargs['embedd_dim'], 'hidden_dim': kwargs['model_dim'],
                'num_classes': kwargs['num_classes'], 'dropout': kwargs['dropout'], 
                'with_tf_idf': kwargs['with_tf_idf']}

    elif model_idx == 'LSTM':
        return {'inp_dim': kwargs['embedd_dim'], 'hidden_dim': kwargs['model_dim'],
                'num_layers': kwargs['num_layers'], 'num_classes': kwargs['num_classes'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}

    elif model_idx == 'CNN':
        return {'num_blocks': kwargs['num_layers'], 'num_classes': kwargs['num_classes'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}

    else: # TRM modules
        return {'inp_dim': kwargs['embedd_dim'], 'model_dim': kwargs['model_dim'],
                'num_layers': kwargs['num_layers'], 'num_classes': kwargs['num_classes'],
                'num_heads': kwargs['num_heads'], 'pwff_dim': 2*kwargs['model_dim'],
                'dropout': kwargs['dropout'], 'with_tf_idf': kwargs['with_tf_idf']}