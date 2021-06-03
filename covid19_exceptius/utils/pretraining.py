from covid19_exceptius.types import *

from torch.nn import Module
import torch


def train_epoch_mlm(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: Module) -> Dict[str, Any]:
    model.train()
    epoch_loss = 0.
    for x, y, m in enumerate(dl):
        predictions = model.forward(x, m)
        loss = loss_fn(predictions, y[m])
        loss.backward()
        optim.step()
        optim.zero_grad()
        epoch_loss += loss.item()
    return {'MLMloss': round(epoch_loss/len(dl), 5)}


@torch.no_grad()
def eval_epoch_mlm(model: Module, dl: DataLoader, loss_fn: Module) -> Dict[str, Any]:
    model.eval()
    epoch_loss = 0.
    for x, y, m in enumerate(dl):
        # forward
        predictions = model.forward(x, m)
        loss = loss_fn(predictions, y[m])
        epoch_loss += loss.item()
    return {'MLMLoss': round(epoch_loss/len(dl), 5)}


class PreTrainer(ABC):
    def __init__(self, 
            model: Module, 
            dls: Tuple[DataLoader, ...],
            optimizer: Optimizer, 
            criterion: Module, 
            target_metric: str,
            print_log: bool = True,
            early_stopping: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_dl, self.dev_dl = dls
        self.logs = {'train': [], 'dev': [], 'test': []}
        self.target_metric = target_metric
        self.trained_epochs = 0
        self.print_log = print_log
        self.early_stop_patience = early_stopping if early_stopping >0 else None

    def iterate(self, num_epochs: int, with_test: Maybe[DataLoader] = None, with_save: Maybe[str] = None) -> Dict[str, Any]:
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

                if with_test is not None:
                    self.logs['test'].append({'epoch': epoch+1, **eval_epoch(self.model, with_test, self.criterion)})

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
        self.logs['train'].append({'epoch': current_epoch, **train_epoch(self.model, self.train_dl, self.optimizer, self.criterion)})
        self.logs['dev'].append({'epoch': current_epoch, **eval_epoch(self.model, self.dev_dl, self.criterion)})
        
        # print if wanted
        if self.print_log:
            print('TRAIN:')
            for k,v in self.logs['train'][-1].items():
                print(f'{k} : {v}')
            print()
            print('DEV:')
            for k,v in self.logs['dev'][-1].items():
                print(f'{k} : {v}')
            print('==' * 72)