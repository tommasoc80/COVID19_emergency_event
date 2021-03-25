import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Linear, Dropout, LayerNorm
import torch.nn.functional as F

from opt_einsum import contract
from typing import Optional, Tuple


def positional_encoding(bs: int, seq_len: int, d_model: int, freq: int=10000, device: str='cpu') -> Tensor:
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, device=device, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) *
                         - (torch.log(torch.tensor(freq, dtype=torch.float, device=device)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.repeat(bs, 1, 1)


class PositionWiseFF(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.ff1 = Linear(in_features=inp_dim, out_features=hidden_dim)
        self.ff2 = Linear(in_features=hidden_dim, out_features=inp_dim)
        self.dropout = Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ff1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        return self.ff2(x) 


class MultiHeadAttention(Module):
    def __init__(self, d_model: int, d_k: int, d_v: int, d_q: int, d_out: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        assert d_model % num_heads == 0 # make sure we can define an attention dim for each head
        self.q_projection = Linear(in_features=d_q, out_features=d_model)
        self.k_projection = Linear(in_features=d_k, out_features=d_model)
        self.v_projection = Linear(in_features=d_v, out_features=d_model)
        self.out_projection = Linear(in_features=d_model, out_features=d_out)
        self.dropout = Dropout(p=dropout)

    def multi_head_attention(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[LongTensor]=None) -> Tensor:
        attn_dim, num_heads = k.shape[-2:]
        dividend = torch.sqrt(torch.tensor(attn_dim, device=k.device, dtype=torch.float))

        weights = contract('bidh,bodh->bioh', q, k) / dividend
        if mask is not None:
            mask = mask.unsqueeze(-1).repeat(1, 1, 1, num_heads)
            weights = weights.masked_fill_(mask == 0, value=-1e10)
        weights = weights.softmax(dim=-2)
        out = contract('bioh,bodh->bidh', weights, v)
        return out.flatten(-2)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[LongTensor]=None) -> Tensor:
        batch_size, seq_len = q.shape[0:2]
        q = self.q_projection(q).view(batch_size, seq_len, -1, self.num_heads)
        v = self.v_projection(v).view(batch_size, seq_len, -1, self.num_heads)
        k = self.k_projection(k).view(batch_size, seq_len, -1, self.num_heads)
        mha = self.multi_head_attention(q, k, v, mask)
        mha = self.dropout(mha)
        out =  self.out_projection(mha)
        return out


class TransformerEncoderLayer(Module):
    def __init__(self, model_dim: int, inp_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        mha_kwargs = {'d_model' : model_dim, 'd_k': inp_dim, 'd_q': inp_dim, 'd_v': inp_dim, 
            'd_out':model_dim, 'num_heads': num_heads, 'dropout': dropout}
        self.mha = MultiHeadAttention(**mha_kwargs)
        self.ff = PositionWiseFF(inp_dim=model_dim, hidden_dim=ff_dim, dropout=dropout)
        self.norm_mha = LayerNorm(normalized_shape=model_dim, eps=1e-12)
        self.norm_ff = LayerNorm(normalized_shape=model_dim, eps=1e-12)
        self.dropout = Dropout(p=dropout)

    def forward(self, inps: Tuple[Tensor, Optional[LongTensor]]) -> Tensor:
        x, mask = inps

        x = self.dropout(x)
        self_attn = self.mha(x, x, x, mask)
        self_attn = self.dropout(x)
        self_attn = x + self_attn
        self_attn = self.norm_mha(self_attn)

        transformed = self.ff(self_attn)
        transformed = self.dropout(transformed)
        transformed = self_attn + transformed
        transformed = self.norm_ff(transformed)

        return transformed, mask


class TransformerEncoder(Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: Tensor) -> Tensor:
        pass

