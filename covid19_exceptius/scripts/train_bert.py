from covid19_exceptius.types import *
from covid19_exceptius.models.bert import make_model, collate_tuples
from covid19_exceptius.utils.training import Trainer
from covid19_exceptius.preprocessing import read_unlabeled, read_labeled

from torch import manual_seed, save
from torch.optim import AdamW
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from warnings import filterwarnings
import sys
import os


SAVE_PREFIX = '/data/s3913171'
LANGS = ['belgium', 'uk', 'poland', 'france', 'italy', 'netherlands',
         'norway', 'hungary']

manual_seed(42)
filterwarnings('ignore')


def sprint(s: str) -> None:
    print(s)
    sys.stdout.flush()


def main(name: str,
       languages: str,
       test_lang: str,
       device: str,
       batch_size: int,
       num_epochs: int,
       weight_decay: float,
       save_path: str,
       kfold: int,
       print_log: bool):

    # concat all lang text for training
    def merge_data(langs: List[str], version: str) -> List[AnnotatedSentence]:
        if version not in ['en', 'or']:
            raise ValueError(f'unknown data version {version}')
        paths = ['./annotations/' + lang + '/full_' + version + '.tsv' for lang in langs]
        return sum([read_labeled(p) for p in paths] ,[])
        
    version = 'en' if 'eng' in name else 'or'
    sprint(f'Model {name}, version {version}')
    if save_path != '':
        save_path = '/'.join([save_path, '_'.join([name, version])]) 
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    else:
        save_path = None

    # get appropriate data  
    languages = LANGS if languages == 'all' else languages
    sprint(f'Training on {languages}...')
    sprint(f'Testing on {test_lang}...') if test_lang != '' else sprint('Not testing...')
    test_ds = None
    if test_lang != '':
        languages.remove(test_lang)
        test_ds = read_unlabeled()

    ds = merge_data(LANGS, version) if languages == 'all' else merge_data(languages, version)

    if not kfold:
        model = make_model(name).to(device)
        
        # random 80%-20% train-dev split
        dev_size = int(.2 * len(ds))
        train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])    
        train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size, 
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=True)
        dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size, 
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=False)
        test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size, shuffle=False,
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device)) if test_ds is not None else None

        optim = AdamW(model.parameters(), lr=3e-05, weight_decay=weight_decay)
        criterion = BCEWithLogitsLoss()
        trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric='accuracy', print_log=print_log)

        best = trainer.iterate(num_epochs, with_test=test_dl, with_save=save_path)
        sprint('Results random split:')
        sprint(f' best dev: {best}')
        sprint(f' best test: {trainer.logs["test"][-1]}')
    
    else:
        # k-fold cross validation
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=42)

        accu = 0.
        for iteration, (train_idces, dev_idces) in enumerate(_kfold.split(ds)):
            model = make_model(name).to(device)

            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size, 
                collate_fn=lambda b: collate_tuples(model, model.tokenizer.pad_token_id, device), shuffle=True)
            dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size, 
                collate_fn=lambda b: collate_tuples(model, model.tokenizer.pad_token_id, device), shuffle=False)
            test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size, shuffle=False,
                collate_fn=lambda b: collate_tuples(model, model.tokenizer.pad_token_id, device)) if test_ds is not None else None

            optim = AdamW(model.parameters(), lr=3e-05, weight_decay=weight_decay)
            criterion = BCEWithLogitsLoss()
            trainer = Trainer(model, (train_dl, dev_dl), optim, criterion, target_metric='event_accuracy', print_log=print_log)
            
            best = trainer.iterate(num_epochs, with_test=test_dl, with_save=save_path)
            sprint(f'Results {kfold}-fold, iteration {iteration}:')
            sprint(f' best dev: {best}')
            sprint(f' best test: {trainer.logs["test"][-1]}')
            accu += trainer.logs["test"][-1]['accuracy']
        sprint(f'Overall accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-l', '--languages', help='what languages to train on (seperated by ,)', type=str, default='all')
    parser.add_argument('-tst', '--test_lang', help='what language to test on', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/COVID-19-event/checkpoints')
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)