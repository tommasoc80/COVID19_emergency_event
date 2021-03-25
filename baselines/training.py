import torch
from torch import Tensor, LongTensor
from torch.nn import Module 
from torch.utils.data import DataLoader
from torch.optim import Optimizer

from typing import Callable, Tuple, Dict

tensor_map = Callable[[Tensor], Tensor]


@torch.no_grad()
def multi_label_metrics(preds: Tensor, labels: LongTensor, thresh: float=0.5) -> Dict[str, float]:
    batch_size, num_classes = preds.shape

    # for evaluation, make all logit predictions >50% to become 1 and other 0
    preds = torch.where(preds.sigmoid()>thresh, torch.ones_like(preds), torch.zeros_like(preds))

    # accuracy score counts the percentage of correctly classified entire sentences
    corrects = preds == labels 
    count_mask = corrects.sum(dim=-1)
    accuracy = count_mask[count_mask==num_classes].shape[0] / batch_size

    # hamming loss counts the fraction of incorrectly classified specific labels
    hamm_loss = corrects[corrects==0].shape[0] / (batch_size * num_classes)

    return {'accuracy':accuracy, 'hamming_loss':hamm_loss}


def train_epoch(model: Module, dl: DataLoader, optim: Optimizer, loss_fn: tensor_map, device: str) -> Tuple[float, float, float]:
    model.train()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y,t) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        
        # forward 
        predictions = model(x, t)
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
def eval_epoch(model: Module, dl: DataLoader, loss_fn: tensor_map, device: str) -> Tuple[float, float, float]:
    model.eval()

    batch_loss, batch_accu, batch_hamm_loss = 0., 0., 0.
    for batch_idx, (x,y,t) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        t = t.to(device)
        predictions = model(x, t)
        loss = loss_fn(predictions, y.float())
        acc_metrics = multi_label_metrics(predictions, y)
        batch_loss += loss.item()
        batch_accu += acc_metrics['accuracy']
        batch_hamm_loss += acc_metrics['hamming_loss']

    return batch_loss/len(dl), batch_accu/len(dl), batch_hamm_loss/len(dl) 