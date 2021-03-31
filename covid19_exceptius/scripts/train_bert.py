from covid19_exceptius.types import *
from covid19_exceptius.models.bert import *
from covid19_exceptius.utils.training import train_epoch, eval_epoch

from torch import manual_seed, save
from torch.optim import AdamW
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from warnings import filterwarnings
import sys
import os

SAVE_PREFIX = '/data/s3913171'


def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


def main(name: str,
       data_path: str,
       test_path: str,
       device: str,
       batch_size: int,
       num_epochs: int,
       weight_decay: float,
       save_path: str,
       kfold: int):

    # an independent function to train once over desired num of epochs
    def train(train_ds: List[AnnotatedSentence], dev_ds: List[AnnotatedSentence], test_ds: Maybe[List[AnnotatedSentence]]) -> Tuple[List[Dict[str, Any]], int]:
        model = make_model(name).to(device)
        criterion = BCEWithLogitsLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=3e-05, weight_decay=weight_decay)

        train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size,
                              collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=True)
        dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size,
                              collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), shuffle=False)
        test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size,
                              collate_fn=lambda batch: collate_tuples(batch, model.tokenizer.pad_token_id), 
                              shuffle=False) if test_ds is not None else None
        
        train_log, dev_log, test_log = [], [], []
        best = 0
        for epoch in range(num_epochs):
            train_log.append(train_epoch(model, train_dl, optimizer, criterion, device))
            sprint(train_log[-1])
            dev_log.append(eval_epoch(model, dev_dl, criterion, device))
            sprint(dev_log[-1])
            sprint('=' * 64)
            if dev_log[-1]['accuracy'] > dev_log[best]['accuracy']:
                best = epoch
                faith = array([c['f1'] for c in dev_log[-1]['column_wise']])
                save(
                    {'faith': faith, 'model_state_dict': model.state_dict()}, f'{save_path}/model.p')
                # eval on test set for each new best model
                if test_ds is not None:
                    test_log.append(eval_epoch(model, test_dl, criterion, device))
                    sprint('\nTEST\n')
                    sprint(test_log[-1])
                    sprint('=' * 64)
        return (train_log, dev_log, test_log), best


    save_path = '/'.join([save_path, name])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    manual_seed(42)
    filterwarnings('ignore')

    if not kfold:
        # 80-20 random train-dev split
        ds = read_labeled(data_path)
        dev_size = int(.2 * len(ds))
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])    
        test_ds = read_labeled(test_path) if test_path != '' else None 
        logs, best = train(train_ds, dev_ds, test_ds)
        sprint('Results random split:')
        sprint(logs[1][best])
    
    else:
        # k-fold cross validation
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=42)
        ds = read_labeled(data_path)
        test_ds = read_labeled(test_path) if test_path != '' else None 

        accu = 0.
        for iteration, (train_idces, dev_idces) in enumerate(_kfold.split(ds)):
            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            logs, best = train(train_ds, dev_ds, test_ds)
            sprint(f'Results {kfold}-fold, iteration {iteration}:')
            sprint(logs[1][best])
            sprint('\n')
            accu += logs[1][best]['accuracy']
        sprint(f'Overall accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-p', '--data_path', help='path to the data tsv', type=str, default='./nlp4ifchallenge/data/english/covid19_disinfo_binary_english_train.tsv')
    parser.add_argument('-tst', '--test_path', help='path to the testing data tsv', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=32)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/nlp4ifchallenge/checkpoints')
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)