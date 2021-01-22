import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, KLDivLoss
from torch.optim import Optimizer, Adam

from sklearn.metrics import accuracy_score, hamming_loss
import pickle

from data_loading import get_datasets, build_dataloaders
from model_library import MultiLabelLSTM

from typing import Callable, Tuple
tensor_map = Callable[[Tensor], Tensor]

# GPU acceleration if applied
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def multi_label_metrics(preds: Tensor, labels: LongTensor, thresh: float=0.5) -> Tuple[float, float]:
    batch_size, num_classes = preds.shape

    # for evaluation, make all label predictions >50% to become 1 and other 0
    preds = torch.where(preds.sigmoid()>thresh, torch.ones_like(preds), torch.zeros_like(preds))

    # accuracy score counts the percentage of correctly classified entire sentences
    corrects = preds == labels 
    count_mask = corrects.sum(dim=-1)
    accuracy = count_mask[count_mask==num_classes].shape[0] / batch_size

    # hamming loss counts the fraction of incorrectly classified specific labels
    hamm_loss = corrects[corrects==0].shape[0] / (batch_size * num_classes)

    return {'accuracy':accuracy, 'hamming_loss':hamm_loss}


def train_epoch(model: MultiLabelLSTM, dl: DataLoader, optim: Optimizer, loss_fn: tensor_map) -> Tuple[float, float, float]:
    model.train()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        
        # forward 
        loss = loss_fn(predictions, y.float())
        acc_metrics = multi_label_metrics(predictions, y)

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        batch_loss += loss.item()
        batch_accu += acc_metrics['accuracy']
        batch_hamm_loss += acc_metrics['hamming_loss']

    return batch_loss/len(dl), batch_accu/len(dl), batch_hamm_loss/len(dl) 


@torch.no_grad()
def eval_epoch(model: MultiLabelLSTM, dl: DataLoader, loss_fn: tensor_map) -> Tuple[float, float, float]:
    model.eval()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        predictions = model(x)
        loss = loss_fn(predictions, y.float())
        acc_metrics = multi_label_metrics(predictions, y)
        batch_loss += loss.item()
        batch_accu += acc_metrics['accuracy']
        batch_hamm_loss += acc_metrics['hamming_loss']

    return batch_loss/len(dl), batch_accu/len(dl), batch_hamm_loss/len(dl) 


def main(path_to_csv_file: str, batch_size: int, num_epochs: int, embedd_dim: int, hidden_dim: int, num_layers: int, num_classes: int,
        learning_rate: float, checkpoint: bool):
    # get loaders with data already tensorized and batched
    prefix = '/'.join(path_to_csv_file.split('/')[:-1])
    csv_name = path_to_csv_file.split('/')[-1].split('.')[0]
    if not checkpoint:
        train_set, test_set = get_datasets(path_to_csv_file)
        pickle.dump((train_set, test_set), open(prefix + '/proc_' + csv_name + '.p', 'wb'))
    else:
        train_set, test_set = pickle.load(open(prefix + '/proc_' + csv_name + '.p', 'rb'))
    train_dl, test_dl = build_dataloaders(train_set, test_set, bs=batch_size)

    # init model
    model = MultiLabelLSTM(embedd_dim, hidden_dim, num_layers, num_classes).to(device)

    # init optimizer and loss functions
    loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)
    # loss_fn = KLDivLoss(reduction='mean').to(device)
    optim = Adam(model.parameters(), lr=learning_rate)

    print('Starting training...')
    for epoch in range(num_epochs):
        train_loss, train_accu, train_hamm_loss = train_epoch(model, train_dl, optim, loss_fn)
        print(f'Epoch {epoch+1}/{num_epochs}: BCE loss={train_loss:.4f}\tSentence accuracy={train_accu*100:2.3f}%\tHamming loss={100*train_hamm_loss:2.3f}%')
        # evaluate every 50 epochs
        if (epoch+1) % 5 == 0:
            val_loss, val_accu, val_hamm_loss = eval_epoch(model, test_dl, loss_fn)
            print('-' * 80)
            print('EVALUATION')
            print(f'BCE loss={val_loss:.4f}\nSentence accuracy={val_accu*100:2.3f}%\nHamming loss={100*val_hamm_loss:2.3f}%')
            print('-' * 80)


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
    parser.add_argument('-lr', '--learning_rate', help='learning rate to use for Adam optimization', type=float, default=1e-03)
    parser.add_argument('--checkpoint', action='store_true', help='whether to skip dataloading', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)