from covid19_exceptius.types import *
from covid19_exceptius.preprocessing import *
from covid19_exceptius.models.bert import make_model, collate_tuples
from covid19_exceptius.utils.training import Trainer

import torch
from torch.optim import AdamW
from torch.nn import Module, BCEWithLogitsLoss
from torch.utils.data import random_split
from sklearn.model_selection import KFold

from warnings import filterwarnings
import sys
import os

SAVE_PREFIX = '/data/s3913171'
LANGS = 'belgium,poland,france,italy,netherlands,norway,hungary,uk'

filterwarnings('ignore')

# reproducability
SEED = torch.manual_seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(42)


def sprint(s: Any) -> None:
    print(s)
    sys.stdout.flush()


def main(name: str,
       languages: List[str],
       test_lang: str,
       device: str,
       batch_size: int,
       num_epochs: int,
       weight_decay: float,
       save_path: str,
       lr: float,
       dropout: float,
       kfold: int,
       max_length: int,
       print_log: bool,
       with_class_weights: bool,
       adaptation: bool,
       load_path: Maybe[str]):
    languages = languages.split(',')

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
        if load_path is not None:
            save_path = save_path + '_pretrain'
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
    else:
        save_path = None
    sprint(f'Full save path: {save_path}')

    # if we want to evaluate language adaptation
    if adaptation:
        # get appropriate data  
        test_ds = None
        if test_lang != '':
            languages.remove(test_lang)
            test_ds = read_labeled('./annotations/' + test_lang + '/full_' + version + '.tsv')
            #test_ds = [AnnotatedSentence(no=s.no, text=s.text[-170:], labels=s.labels) for s in test_ds]
        sprint(f'Training on {languages}...')
        sprint(f'Testing on {test_lang}...') if test_lang != '' else sprint('Not testing...')

        ds = merge_data(languages, version)
        #ds = [AnnotatedSentence(no=s.no, text=s.text[-170:], labels=s.labels) for s in ds]

    # using standard train-dev-test splits
    else:
        dss = {}
        dss['train'] = read_labeled('annotations/all_train_or.tsv')
        dss['dev'] = read_labeled('annotations/all_dev_or.tsv')
        dss['test'] = None
        if test_lang != '':
            dss['test'] = read_labeled('./annotations/' + test_lang + '/test_' + version + '.tsv')
        
        #dss = {k: [AnnotatedSentence(no=s.no, text=s.text[-170:], labels=s.labels) for s in v] if v is not None else None for k, v in dss.items()}
        sprint(f'Training on train splits of all languages...')
        sprint(f'Testing on {test_lang}...') if test_lang != '' else sprint('Not testing...')


    if not kfold:
        model = make_model(name, version='classifier', max_length=max_length, dropout_rate=dropout).to(device)
        if load_path is not None:
            model.load_core(load_path)
        
        if adaptation:
            # random 80%-20% train-dev split
            dev_size = int(.2 * len(ds))
            train_ds, dev_ds = random_split(ds, [len(ds) - dev_size, dev_size])  
        
        else:
            train_ds, dev_ds, test_ds = dss['train'], dss['dev'], dss['test']

        train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size, worker_init_fn=SEED,
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=True)
        dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size, worker_init_fn=SEED,
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=False)
        test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size, shuffle=False, worker_init_fn=SEED,
            collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device)) if test_ds is not None else None

        optim = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        #class_weights = torch.tensor(extract_class_weights(train_ds), dtype=longt, device=device)
        criterion = BCEWithLogitsLoss() if not with_class_weights else BCEWithLogitsLoss(pos_weight=class_weights)
        trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric='event_accuracy_label', print_log=print_log)

        best = trainer.iterate(num_epochs, with_save=save_path)
        sprint('Results random split:')
        sprint(f' best dev: {best}')
        if test_dl is not None:
            sprint(f' best test: {trainer.logs["test"][-1]}')
    
    else:
        # k-fold cross validation
        _kfold = KFold(n_splits=kfold, shuffle=True, random_state=42)

        accu = 0.
        for iteration, (train_idces, dev_idces) in enumerate(_kfold.split(ds)):
            model = make_model(name, version='classifier', max_length=max_length).to(device)
            if load_path is not None:
                model.load_core(load_path)

            train_ds = [s for i, s in enumerate(ds) if i in train_idces]
            dev_ds = [s for i, s in enumerate(ds) if i in dev_idces]
            train_dl = DataLoader(model.tensorize_labeled(train_ds), batch_size=batch_size, worker_init_fn=SEED,
                collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=True)
            dev_dl = DataLoader(model.tensorize_labeled(dev_ds), batch_size=batch_size, worker_init_fn=SEED,
                collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device), shuffle=False)
            test_dl = DataLoader(model.tensorize_labeled(test_ds), batch_size=batch_size, shuffle=False, worker_init_fn=SEED,
                collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device)) if test_ds is not None else None

            optim = AdamW(model.parameters(), lr=3e-05, weight_decay=weight_decay)
            #class_weights = torch.tensor(extract_class_weights(train_ds), dtype=longt, device=device)
            criterion = BCEWithLogitsLoss() if not with_class_weights else BCEWithLogitsLoss(pos_weight=class_weights)
            trainer = Trainer(model, (train_dl, dev_dl, test_dl), optim, criterion, target_metric='event_accuracy_label', print_log=print_log)
            
            best = trainer.iterate(num_epochs, with_save=save_path)
            sprint(f'Results {kfold}-fold, iteration {iteration + 1}:')
            sprint(f' best dev: {best}')
            if test_dl is not None:
                sprint(f' best test: {trainer.logs["test"][-1]}')
                accu += trainer.logs["test"][-1]['event_accuracy_label']
        sprint(f'Overall test accuracy {kfold}-fold: {accu/kfold}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-lang', '--languages', help='what languages to train on (separated by ,)', type=str, default=LANGS)
    parser.add_argument('-tst', '--test_lang', help='what language to test on', type=str, default='')
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-lr', '--lr', help='learning rate to use for optimization', type=float, default=3e-05)
    parser.add_argument('-dr', '--dropout', help='dropout rate in classifier layer', type=float, default=0.5)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=f'{SAVE_PREFIX}/COVID-19-event/checkpoints')
    parser.add_argument('-l', '--load_path', help='where to load pretrained core from (default no)', type=str, default=None)
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-len', '--max_length', help='truncate to maximum sentence length', type=int, default=256)
    parser.add_argument('-kfold', '--kfold', help='k-fold cross validation', type=int, default=0)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('--with_class_weights', action='store_true', help='compute class weights for loss penalization', default=False)
    parser.add_argument('--adaptation', action='store_true', help='do language adaptation experiment', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)