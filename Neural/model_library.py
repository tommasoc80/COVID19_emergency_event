import torch
from torch import Tensor, LongTensor
from torch.nn import Module, LSTM, Linear


class MultiLabelLSTM(Module):
    def __init__(self, inp_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super().__init__()
        self.num_layers = num_layers
        self.lstm = LSTM(input_size=inp_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc = Linear(in_features=2*hidden_dim, out_features=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        hiddens, _ = self.lstm(x)
        context = torch.cat((hiddens[:,-1,:self.lstm.hidden_size], hiddens[:,0,self.lstm.hidden_size:]), dim=-1)
        out = self.fc(context)
        return out