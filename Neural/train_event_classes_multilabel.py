import torch
from torch import Tensor, LongTensor
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam

import pickle
from sklearn.model_selection import KFold

from data_loading import vectorize_dataset, build_dataloaders, EventClassesMultilabel
from training import train_epoch, eval_epoch
from model_library import models_dict, setup_model_kwargs

from typing import Tuple, Optional

# GPU acceleration if applied
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_kappa = 10 # 10-fold cross-validation]


def main(path_to_csv_file: str, 
        model_idx: str,
        batch_size: int, 
        num_epochs: int, 
        embedd_dim: int, 
        hidden_dim: int, 
        num_layers: int, 
        num_classes: int,
        learning_rate: float, 
        dropout: float, 
        weight_decay: float, 
        early_stop_patience: int,
        embeddings: str,
        dev_freq: int, 
        k_fold: bool, 
        checkpoint: bool,
        num_heads: Optional[int]=None):

    # a function for training a fresh model into a given split with given hyper-params
    def train_full(train_dl: DataLoader, test_dl: DataLoader) -> Tuple[float, float, float]:
        # init appropriate model
        model_kwargs = {'embedd_dim': embedd_dim, 'model_dim': hidden_dim, 'num_classes': num_classes, 'num_layers': num_layers,
                        'dropout': dropout, 'num_heads': num_heads}
        model = models_dict[model_idx](**setup_model_kwargs(model_idx, model_kwargs)).to(device)

        # init optimizer and loss function
        loss_fn = BCEWithLogitsLoss(reduction='mean').to(device)
        optim = Adam(model.parameters(), lr=learning_rate, betas=[.9, .999], weight_decay=weight_decay)

        early_stop_pointer, patience = 0., early_stop_patience
        metrics = (None, None, None)

        for epoch in range(num_epochs):
            train_loss, train_accu, train_hamm_loss = train_epoch(model, train_dl, optim, loss_fn, device)
            print(f'Epoch {epoch+1}/{num_epochs}: BCE loss={train_loss:.4f}\tSentence accuracy={train_accu*100:2.3f}%\tHamming loss={100*train_hamm_loss:2.3f}%')
           
            # evaluate every some epochs
            if (epoch+1) % dev_freq == 0:
                val_loss, val_accu, val_hamm_loss = eval_epoch(model, test_dl, loss_fn, device)
                print('-' * 80)
                print('EVALUATION:')
                print(f'BCE loss={val_loss:.4f}\nSentence accuracy={val_accu*100:2.3f}%\nHamming loss={100*val_hamm_loss:2.3f}%')
                print('-' * 80)

                # stop if test accuracy is falling for 4 evaluations
                if val_accu < early_stop_pointer:
                    patience -= 1
                    if not  patience:
                        print('\nEarly stopping.')
                        break
                else:
                    patience = early_stop_patience
                    early_stop_pointer = val_accu
                    metrics = (val_loss, val_accu, val_hamm_loss)
        return metrics


    # parse path and csv file name
    prefix = '/'.join(path_to_csv_file.split('/')[:-1])
    csv_name = path_to_csv_file.split('/')[-1].split('.')[0]   

    # skip the data vectorization if already run once
    if not checkpoint:
        dataset = vectorize_dataset(path_to_csv_file, embeddings=embeddings)
        pickle.dump(dataset, open(prefix + '/vectors_' + embeddings + '_' + csv_name + '.p', 'wb'))
    else:
        dataset = pickle.load(open(prefix + '/vectors_' + embeddings + '_' + csv_name + '.p', 'rb'))

    if k_fold:
        # save average scores for k-fold cross validation
        bce_loss, sent_accu, hamm_loss = 0., 0., 0.
        kf = KFold(n_splits=_kappa, shuffle=True, random_state=42)
        text, labels = zip(*dataset)
        for iteration, (train_idces, test_idces) in enumerate(kf.split(text)):
            train_set = [dataset[idx] for idx in range(len(dataset)) if idx in list(train_idces)]
            test_set = [dataset[idx] for idx in range(len(dataset)) if idx in list(test_idces)]
            train_dl, test_dl = build_dataloaders(train_set, test_set, bs=batch_size)
            print('='*80)
            print(f'Evaluating on fold {iteration+1} from {_kappa}:')
            print('='*80)
            metrics = train_full(train_dl, test_dl)

            # update scores
            bce_loss += metrics[0]
            sent_accu += metrics[1]
            hamm_loss += metrics[2]

        print(f'\nResults from {_kappa}-fold cross validation:')
        print(f'BCE Loss={bce_loss/_kappa:.4f}\nSentence accuracy={100*(sent_accu/_kappa):2.3f}%\nHamming loss={100*(hamm_loss/_kappa):2.3f}%')
    

    else:
        # run once for a generated random split
        train_set, test_set = dataset.random_train_test_split()
        train_dl, test_dl = build_dataloaders(train_set, test_set, bs=batch_size)
        print('='*80)
        print(f'Training on random train-test split:')
        print('='*80)
        metrics = train_full(train_dl, test_dl)
        print(f'\nResults from random train-test split:')
        print(f'BCE Loss={metrics[0]:.4f}\nSentence accuracy={100*metrics[1]:2.3f}%\nHamming loss={100*metrics[2]:2.3f}%')

    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path_to_csv_file', help='path to CSV file containing the data', type=str)
    parser.add_argument('-m', '--model_idx', help='which model to use for training (see code for options)', type=str, default='LSTM')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=32)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=100)
    parser.add_argument('-de', '--embedd_dim', help='dimensionality of input word vectors', type=int, default=300)
    parser.add_argument('-dh', '--hidden_dim', help='dimensionality of models hidden vectors', type=int, default=150)
    parser.add_argument('-l', '--num_layers', help='depth of model', type=int, default=1)
    parser.add_argument('-H', '--num_heads', help='number of multi-attention heads for transformers', type=int, default=None)
    parser.add_argument('-c', '--num_classes', help='num of target classes', type=int, default=8)
    parser.add_argument('-lr', '--learning_rate', help='learning rate to use for Adam optimization', type=float, default=.005)
    parser.add_argument('-dr', '--dropout', help='dropout rate used for regularization', type=float, default=.0)
    parser.add_argument('-wd', '--weight_decay', help='weight decay rate used for regularization', type=float, default=.0)
    parser.add_argument('-emb', '--embeddings', help='what type of embedder to use for data vectorization (see code)', type=str, default='glove')
    parser.add_argument('-early', '--early_stop_patience', help='stop training if evaluation is worse in that many epochs', type=int, default=6)
    parser.add_argument('-dev', '--dev_freq', help='every how many epochs to evaluate', type=int, default=5)
    parser.add_argument('--k_fold', action='store_true', help='whether to evaluate with k-fold cross validation', default=False)
    parser.add_argument('--checkpoint', action='store_true', help='whether to skip dataloading', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)