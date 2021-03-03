import torch
from torch import Tensor, LongTensor
from torch.nn import Module, Linear, Dropout, BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW, Optimizer

from transformers import BertTokenizer, BertModel
from sklearn.model_selection import KFold, train_test_split

from paragraph_similarity import denoise_text
from data_loading import vectorize_label, array
from training import multi_label_metrics

import sys
from typing import Tuple, List, Callable

Sentence = List[int]
Sample = Tuple[Sentence, array]
Samples = List[Sample]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def sprint(s: str):
    print(s)
    sys.stdout.flush()


class BertForMultiLabelSequenceClassification(Module):
    def __init__(self, num_classes: int, core_path: str, model_dim: int=768, dropout: float=0.1):
        super().__init__()
        self.core = BertModel.from_pretrained(core_path)
        self.cls = Linear(model_dim, num_classes)
        self.dropout = Dropout(p=dropout)

    def forward(self, word_ids: Tensor, attention_mask: Tensor) -> Tensor:
        pooler = self.core(word_ids, attention_mask).pooler_output
        pooler = self.dropout(pooler)
        out =  self.cls(pooler)
        return out 


def load_data(csv_path: str, tokenizer: BertTokenizer) -> Samples:
    text, labels = zip(*pd.read_csv(csv_path).values.tolist())
    
    # remove paragraph indexing in the start of sentences
    text = denoise_text(text)

    # tokenize input sentences
    tokens = [tokenizer.encode(sent, add_special_tokens=True) for sent in text]
    
    # vectorize labels  
    labels = [vectorize_label(l) for l in labels]

    # return dataset
    return list(zip(tokens, labels))


def collator(padding_value: int) -> Callable[[Samples], Tuple[LongTensor, LongTensor]]:
    def collate_fn(batch: Samples) -> Tuple[LongTensor, LongTensor]:
        xs, ys = zip(*batch)
        xs = pad_sequence([LongTensor(x) for x in xs], batch_first=True, padding_value=padding_value)
        ys = torch.stack([LongTensor(y) for y in ys], dim=0)
        return xs, ys
    return collate_fn


def train_epoch(model: BertForMultiLabelSequenceClassification, 
            loader: DataLoader, 
            optim: Optimizer, 
            loss_fn: Module,
            word_pad_id: int) -> Tuple[float, float, float]:
    model.train()
    bce_loss, accu, hamm_loss = 0., 0., 0.
    for x, y in loader:
        # forward
        x = x.to(device)
        y = y.to(device)
        mask = torch.where(x.eq(word_pad_id), 0, 1)
        preds = model.forward(x, mask)
        loss = loss_fn(preds, y.float())
        accs = multi_label_metrics(preds, y)

        # back-prop
        loss.backward()
        optim.step()
        optim.zero_grad()

        # update metrics
        bce_loss += loss.item()
        accu += accs['accuracy']
        hamm_loss += accs['hamming_loss']
    
    return bce_loss/len(loader), accu/len(loader), hamm_loss/len(loader)


@torch.no_grad
def eval_epoch(model: BertForMultiLabelSequenceClassification, 
            loader: DataLoader, 
            loss_fn: Module,
            word_pad_id: int) -> Tuple[float, float, float]:
    model.eval()
    bce_loss, accu, hamm_loss = 0., 0., 0.
    for x, y in loader:
        # forward
        x = x.to(device)
        y = y.to(device)
        mask = torch.where(x.eq(word_pad_id), 0, 1)
        preds = model.forward(x, mask)
        loss = loss_fn(preds, y.float())
        accs = multi_label_metrics(preds, y)

        # update metrics
        bce_loss += loss.item()
        accu += accs['accuracy']
        hamm_loss += accs['hamming_loss']
    
    return bce_loss/len(loader), accu/len(loader), hamm_loss/len(loader)


def main(csv_path: str,
         batch_size: int,
         learning_rate: float,
         weight_decay: float,
         dropout: float,
         num_epochs: int,
         ):
    # make and split data from csv
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    ds = load_data(csv_path, tokenizer)
    X_train, X_test, y_train, y_test, train_idces, test_idces = train_test_split([s[0] for s in ds], [s[1] for s in ds], 
        np.arange(len(ds)), test_size=0.20, random_state=42)

    # build dataloaders
    train_dl = DataLoader(list(zip(X_train, y_train)), shuffle=True, batch_size=batch_size, collate_fn=collator(tokenizer.pad_token_id))
    test_dl = DataLoader(list(zip(X_test, y_test)), shuffle=False, batch_size=batch_size, collate_fn=collator(tokenizer.pad_token_id))

    # init model, optimizer and loss funciton
    model = BertForMultiLabelSequenceClassification(num_classes=8, core_path='bert-base-uncased', dropout=dropout).to(device)
    optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = BCEWithLogitsLoss().to(device)

    sprint('Training...')
    for epoch in range(num_epochs):
        train_loss, train_accu, train_hl = train_epoch(model, train_dl, optim, loss_fn, tokenizer.pad_token_id)
        val_loss, val_accu, val_hl = eval_epoch(model, test_dl, loss_fn, tokenizer.pad_token_id)
        sprint(f'Epoch {epoch+1}/{num_epochs}: BCE loss={train_loss:.4f}, Accuracy={train_accu*100:2.3f}, Hamming loss={train_hl*100:2.3f}')
        sprint(f'Evaluation----> BCE loss={val_loss:.4f}, Accuracy={val_accu*100:2.3f}, Hamming loss={val_hl*100:2.3f}')
        sprint()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_csv_file', help='path to CSV file containing the data', type=str)
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=4)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=30)
    parser.add_argument('-c', '--num_classes', help='num of target classes', type=int, default=8)
    parser.add_argument('-lr', '--learning_rate', help='learning rate to use for Adam optimization', type=float, default=3e-05)
    parser.add_argument('-dr', '--dropout', help='dropout rate used for regularization', type=float, default=.1)
    parser.add_argument('-wd', '--weight_decay', help='weight decay rate used for regularization', type=float, default=.01)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)