from covid19_exceptius.types import *
from covid19_exceptius.models.bagging import *
from covid19_exceptius.preprocessing import read_labeled
from covid19_exceptius.utils.embeddings import WordEmbedder, make_word_embedder
from covid19_exceptius.utils.training import train_epoch_supervised, eval_epoch_supervised
from covid19_exceptius.utils.tf_idf import TfIdfTransform, extract_tf_idfs

from torch import manual_seed, save, tensor
from torch.optim import AdamW
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from warnings import filterwarnings
import sys
import os

SAVE_PREFIX = '.'
train_epoch = train_epoch_supervised
eval_epoch = eval_epoch_supervised

def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


def main(embeddings: str,
        with_tf_idf: int,
        aggregator: str,
        inp_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        data_path: str,
        test_path: str,
        device: str,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        weight_decay: float,
        save_path: str,
        kfold: int):

    # an independent function to train once over desired num of epochs
    def train(train_ds: List[AnnotatedSentence], dev_ds: List[AnnotatedSentence], test_ds: Maybe[List[AnnotatedSentence]]) -> Tuple[List[Dict[str, Any]], int]:
        
        model_kwargs = {'aggregator': aggregator, 'inp_dim': inp_dim, 'hidden_dim': hidden_dim, 
            'num_classes': num_classes, 'dropout': dropout, 'with_tf_idf': with_tf_idf}
        model = make_model(model_kwargs).to(device)
        criterion = BCEWithLogitsLoss().to(device)
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        we = make_word_embedder(embeddings)
        tf_idf_transform = None
        if with_tf_idf:
            _, tf_idf_transform = extract_tf_idfs([s.text for s in train_ds], lsa_components=with_tf_idf)

        train_dl = DataLoader(tensorize_labeled(train_ds, we, tf_idf_transform), batch_size=batch_size,
                              collate_fn=collate_tuples, shuffle=True)
        dev_dl = DataLoader(tensorize_labeled(dev_ds, we, tf_idf_transform), batch_size=batch_size,
                              collate_fn=collate_tuples, shuffle=False)
        test_dl = DataLoader(tensorize_labeled(test_ds, we, tf_idf_transform), batch_size=batch_size,
                              collate_fn=collate_tuples, shuffle=False) if test_ds is not None else None
        
        train_log, dev_log, test_log = [], [], []
        best = 0
        for epoch in range(num_epochs):
            train_log.append(train_epoch(model, train_dl, optimizer, criterion))
            sprint(train_log[-1])
            dev_log.append(eval_epoch(model, dev_dl, criterion))
            sprint(dev_log[-1])
            sprint('=' * 64)
            if dev_log[-1]['mean_f1'] > dev_log[best]['mean_f1']:
                best = epoch
                if test_ds is not None:
                    test_log.append(eval_epoch(model, test_dl, criterion))
                    sprint('\nTEST\n')
                    sprint(test_log[-1])
                    sprint('=' * 64)
                else:
                    test_log.append('empty')
        return (train_log, dev_log, test_log), best


    save_path = '/'.join([save_path, 
        '_'.join([embeddings, 'tf' + str(with_tf_idf), 'H' + str(hidden_dim)])])
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    manual_seed(42)
    filterwarnings('ignore')

    if not kfold:
        # 80-20 random train-dev split
        train_ds, dev_ds = read_labeled(data_path + '/all_train_or.tsv'), read_labeled(data_path + '/all_dev_or.tsv')
        test_ds = read_labeled(test_path) if test_path != '' else None 
        logs, best = train(train_ds, dev_ds, test_ds)
        sprint('Results random split:')
        sprint(f' best dev: {logs[1][best]}')
        sprint(f' best test: {logs[2][best]}')
    
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
            sprint(f'Results {kfold}-fold, iteration {iteration+1}:')
            sprint(logs[1][best])
            sprint('\n')
            accu += logs[1][best]['accuracy']
        sprint(f'Overall accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-emb', '--embeddings', help='what type of word embeddings to use', type=str, default='glove_lg')
    parser.add_argument('-tf', '--with_tf_idf', help='size of tf-idf components to use (0 for no tf-idfs)', type=int, default=100)
    parser.add_argument('-agg', '--aggregator', help='aggregation method (Boe or RNN)', type=str)
    parser.add_argument('-di', '--inp_dim', help='embedding size', type=int, default=300)
    parser.add_argument('-dh', '--hidden_dim', help='size of MLP hidden layer (0 to ommit)', type=int, default=128)
    parser.add_argument('-c', '--num_classes', help='number of target classes', type=int, default=8)
    parser.add_argument('-p', '--data_path', help='path to the data tsv', type=str)
    parser.add_argument('-tst', '--test_path', help='path to the testing data tsv', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=100)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=SAVE_PREFIX)
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('-dr', '--dropout', help='model dropout to use in training', type=float, default=0.25)
    parser.add_argument('-lr', '--learning_rate', help='learning rate to use in optimizer', type=float, default=1e-03)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)