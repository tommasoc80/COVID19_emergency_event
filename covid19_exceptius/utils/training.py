from covid19_exceptius.types import *
from covid19_exceptius.utils.metrics import get_metrics

from torch.nn import Module
import torch
import sys


def train_epoch_mlm(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module) -> Dict[str, Any]:
    model.train()
    epoch_loss = 0.
    for x, y, m in dl:
        predictions = model.forward(x, m)
        loss = loss_fn(predictions, y[m==1])
        loss.backward()
        optim.step()
        optim.zero_grad()
        epoch_loss += loss.item()
    return {'MLMloss': -round(epoch_loss/len(dl), 5)}


@torch.no_grad()
def eval_epoch_mlm(model: Module, dl: DataLoader, loss_fn: Module) -> Dict[str, Any]:
    model.eval()
    epoch_loss = 0.
    for x, y, m in dl:
        # forward
        predictions = model.forward(x, m)
        loss = loss_fn(predictions, y[m==1])
        epoch_loss += loss.item()
    return {'MLMLoss': -round(epoch_loss/len(dl), 5)}


def train_epoch_supervised(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module) -> Dict[str, Any]:
    model.train()

    epoch_loss = 0.
    all_preds:  List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
        # forward 
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())

        # backprop
        loss.backward()
        optim.step()
        optim.zero_grad()

        all_preds.extend(predictions.sigmoid().round().cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        epoch_loss += loss.item()

    return {'BCEloss': -round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}


@torch.no_grad()
def eval_epoch_supervised(model: Module, dl: DataLoader, loss_fn: Module) -> Dict[str, Any]:
    model.eval()

    epoch_loss = 0.
    all_preds: List[List[int]] = []
    all_labels: List[List[int]] = []

    for batch_idx, (x, y) in enumerate(dl):
        # forward
        predictions = model.forward(x)
        loss = loss_fn(predictions, y.float())

        all_preds.extend(predictions.sigmoid().round().cpu().tolist())
        all_labels.extend(y.cpu().tolist())

        epoch_loss += loss.item()

    return {'BCEloss': -round(epoch_loss/len(dl), 5), **get_metrics(all_preds, all_labels)}


class Trainer(ABC):
    def __init__(self, 
            model: Module, 
            dls: Tuple[Maybe[DataLoader], ...],
            optimizer: Optimizer, 
            criterion: Module, 
            target_metric: str,
            print_log: bool = True,
            early_stopping: int = 0,
            pretrain: bool = False):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl, self.dev_dl, self.test_dl = dls
        self.logs = {'train': [], 'dev': [], 'test': []}
        self.target_metric = target_metric
        self.trained_epochs = 0
        self.print_log = print_log
        self.early_stop_patience = early_stopping if early_stopping >0 else None
        self.train_fn = train_epoch_mlm if pretrain else train_epoch_supervised
        self.eval_fn = eval_epoch_mlm if pretrain else eval_epoch_supervised

    def iterate(self, num_epochs: int, with_save: Maybe[str] = None) -> Dict[str, Any]:
        best = {self.target_metric: 0.}
        patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs
        for epoch in range(num_epochs):
            self.step()
  
            # update logger for best - save - test - early stopping
            if self.logs['dev'][-1][self.target_metric] > best[self.target_metric]:
                best = self.logs['dev'][-1]
                patience = self.early_stop_patience if self.early_stop_patience is not None else num_epochs

                if with_save is not None:
                    torch.save(self.model.state_dict(), with_save)

                if self.test_dl is not None:
                    self.logs['test'].append({'epoch': epoch+1, **self.eval_fn(self.model, self.test_dl, self.criterion)})

            else:
                patience -= 1
                if not patience:
                    self.trained_epochs += epoch + 1
                    break
        self.trained_epochs += num_epochs
        return best

    def step(self):
        current_epoch = len(self.logs['train']) + 1

        # train - eval this epoch
        self.logs['train'].append({'epoch': current_epoch, **self.train_epoch()})
        self.logs['dev'].append({'epoch': current_epoch, **self.eval_epoch()})
        
        # print if wanted
        if self.print_log:
            print('TRAIN:')
            for k,v in self.logs['train'][-1].items():
                sprint(f'{k} : {v}')
            print()
            print('DEV:')
            for k,v in self.logs['dev'][-1].items():
                sprint(f'{k} : {v}')
            print('==' * 72)
            sys.stdout.flush()

    def train_epoch(self):
        return self.train_fn(self.model, self.train_dl, self.optimizer, self.criterion)

    def eval_epoch(self):
        return self.eval_fn(self.model, self.dev_dl, self.criterion)