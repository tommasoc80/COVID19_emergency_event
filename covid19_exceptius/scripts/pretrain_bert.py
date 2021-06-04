from covid19_exceptius.types import *
from covid19_exceptius.preprocessing import read_tokenized
from covid19_exceptius.models.bert import make_mlm_model, collate_with_mask
from covid19_exceptius.utils.training import Trainer

from torch import manual_seed
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

from warnings import filterwarnings
import sys
import os

DATA_ROOT = '/data/s3913171/COVID-19-event/CORPUS_EXCEPTIUS'

SEED = manual_seed(42)
filterwarnings('ignore')


def sprint(s: Any) -> None:
    print(s)
    sys.stdout.flush()


def main(name: str,
         data_root: str,
         device: str,
         num_epochs: int,
         batch_size: int,
         learning_rate: float,
         weight_decay: float,
         max_length: int,
         print_log: bool,
         early_stopping: int,
         checkpoint: bool,
         save_path: Maybe[str]):

    sprint('Loading model...')
    model = make_mlm_model(name, max_length=max_length).to(device)

    # get data from checkpoint if wanted
    sprint('Preparing data...')
    if not checkpoint:
        from covid19_exceptius.models.bert import make_pretrain_dataset

        train_ds = make_pretrain_dataset(os.path.join(data_root, 'train'), tokenizer=model.tokenizer, truncation=True, 
                                        max_length=max_length, with_save=os.path.join(data_root, 'train', 'full.txt'))
        dev_ds = make_pretrain_dataset(os.path.join(data_root, 'dev'), tokenizer=model.tokenizer, truncation=True, 
                                        max_length=max_length, with_save=os.path.join(data_root, 'dev', 'full.txt'))
        test_ds = make_pretrain_dataset(os.path.join(data_root, 'test'), tokenizer=model.tokenizer, truncation=True, 
                                        max_length=max_length, with_save=os.path.join(data_root, 'test', 'full.txt'))
    else:
        train_ds = read_tokenized(os.path.join(data_root, 'train', 'full.txt'))
        dev_ds = read_tokenized(os.path.join(data_root, 'dev', 'full.txt'))
        test_ds = read_tokenized(os.path.join(data_root, 'test', 'full.txt'))

    pad_id = model.tokenizer.pad_token_id
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED,
                          collate_fn = lambda b: collate_with_mask(b, mask_fn=model.mask, padding_value=pad_id, device=device))
    dev_dl = DataLoader(dev_ds, shuffle=False, batch_size=batch_size, worker_init_fn=SEED,
                          collate_fn = lambda b: collate_with_mask(b, mask_fn=model.mask, padding_value=pad_id, device=device))
    test_dl = DataLoader(test_ds, shuffle=True, batch_size=batch_size, worker_init_fn=SEED,
                          collate_fn = lambda b: collate_with_mask(b, mask_fn=model.mask, padding_value=pad_id, device=device))

    optim = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = CrossEntropyLoss(reduction='mean', ignore_index=pad_id).to(device)

    sprint(f'Training for {num_epochs} epochs...')
    trainer = Trainer(model, [train_dl, dev_dl, test_dl], optim, criterion, target_metric='MLMLoss', print_log=print_log,
                      early_stopping=early_stopping, pretrain=True)

    best = trainer.iterate(num_epochs, with_save=save_path)
    sprint(f'Results best dev set: {best}')
    sprint(f'Results test set: {trainer.logs["test"][-1]}')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str)
    parser.add_argument('-r', '--data_root', help='path to train-dev-test splits', type=str, default=DATA_ROOT)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-bs', '--batch_size', help='batch size to use for training', type=int, default=16)
    parser.add_argument('-e', '--num_epochs', help='how many epochs of training', type=int, default=7)
    parser.add_argument('-s', '--save_path', help='where to save best model', type=str, default=None)
    parser.add_argument('-early', '--early_stopping', help='early stopping patience (default no)', type=int, default=0)
    parser.add_argument('-lr', '--learning_rate', help='learning rate to use for optimization', type=float, default=1e-03)
    parser.add_argument('-wd', '--weight_decay', help='weight decay to use for regularization', type=float, default=1e-02)
    parser.add_argument('-len', '--max_length', help='truncate to maximum sentence length', type=int, default=256)
    parser.add_argument('--print_log', action='store_true', help='print training logs', default=False)
    parser.add_argument('--checkpoint', action='store_true', help='load already tokenized data from checkpoint', default=False)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)