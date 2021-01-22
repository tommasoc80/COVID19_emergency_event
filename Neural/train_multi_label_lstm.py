import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Optimizer, AdamW

from data_loading import get_dataloaders
from model_library import MultiLabelLSTM

from typing import Callable, Tuple
tensor_map = Callable[[Tensor], Tensor]

# GPU acceleration if applied
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def multi_label_accuracy(preds, labels, thresh: float=0.5):
    preds[preds>thresh] = 1
    preds[preds<thresh] = 0


def train_epoch(model: MultiLabelLSTM, dl: DataLoader, optim: Optimizer, loss_fn: tensor_map) -> Tuple[float, float]:
    model.train()

    batch_loss, batch_accu = 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        loss = loss_fn(predictions, y.float())

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        batch_loss += loss.item()
        batch_accu += 0.

    return batch_loss/len(dl), batch_accu/len(dl) 


@torch.no_grad()
def eval_epoch(model: MultiLabelLSTM, dl: DataLoader, optim: Optimizer, loss_fn: tensor_map) -> Tuple[float, float]:
    model.eval()

    batch_loss, batch_accu = 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        loss = loss_fn(predictions, y.float())
        batch_loss += loss.item()
        batch_accu += 0.

    return batch_loss/len(dl), batch_accu/len(dl) 


def main(path_to_csv_file: str, batch_size: int, num_epochs: int, embedd_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
    # get loaders with data already tensorized and batched
    train_dl, test_dl = get_dataloaders(path_to_csv_file, bs=batch_size)

    # init model
    model = MultiLabelLSTM(embedd_dim, hidden_dim, num_layers, num_classes).to(device)

    # init optimizer and loss functions
    loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)
    optim = AdamW(model.parameters())

    print('Starting training...')
    for epoch in range(num_epochs):
        train_loss, train_accu = train_epoch(model, train_dl, optim, loss_fn)
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train loss={train_loss:.4f}\tTrain accuracy={train_accu*100:2.3f}%')
        # evaluate every 50 epochs
        if (epoch+1) % 50 == 0:
            val_loss, val_accu = eval_epoch(model, test_dl, loss_fn)
            print(f'Train loss={val_loss:.4f}\tTrain accuracy={val_accu*100:2.3f}%')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_csv_file', help='path to CSV file containing the data', type=str)
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=32)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=500)
    parser.add_argument('-de', '--embedd_dim', help='Dimensionality of input word vectors', type=int, default=300)
    parser.add_argument('-dh', '--hidden_dim', help='dimensionality of LSTM hidden vectors', type=int, default=150)
    parser.add_argument('-l', '--num_layers', help='depth of LSTM model', type=int, default=1)
    parser.add_argument('-c', '--num_classes', help='num of target classes', type=int, default=8)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)