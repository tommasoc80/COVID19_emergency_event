from covid19_exceptius.types import *
from covid19_exceptius.models.bert import make_model, collate_tuples
from covid19_exceptius.utils.training import eval_epoch_supervised
from covid19_exceptius.preprocessing import read_labeled

import torch
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
import os


ROOT = '/home/s3913171/COVID19_emergency_event/annotations'


def main(name: str,
         data_root: str,
         max_length: int,
         load_path: str,
         device: str,
         save_path: Maybe[str],
         test_only_one: Maybe[str]
        ):

    # load pretrained model
    print('Loading model...')
    model = make_model(name, version='classifier', max_length=max_length).eval().to(device)
    model.load_state_dict(torch.load(load_path))
    criterion = BCEWithLogitsLoss().to(device)


    # load data for all countries
    countries = [f for f in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, f))]
    test_files = [[os.path.join(data_root, country, f) for f in os.listdir(os.path.join(data_root, country)) if f.startswith('test_or')][0]
                    for country in countries]

    # do only one country if desired
    if test_only_one is not None:
        index = countries.index(test_only_one)
        countries, test_files = countries[index], test_files[index]

    results = {}
    print(f'Testing on {len(countries)} countries...')
    for i, file in enumerate(tqdm(test_files)):
        ds = read_labeled(file)
        dl = DataLoader(model.tensorize_labeled(ds), shuffle=False, batch_size=16, 
                        collate_fn=lambda b: collate_tuples(b, model.tokenizer.pad_token_id, device))
        results[countries[i]] = eval_epoch_supervised(model, dl, criterion)

    print(f'Results per-language evaluation:')
    for country, metrics in results.items():
        print(f' {country} : ')
        for _k, _v in metrics.items():
            print(f' {_k} = {_v}')
        print()
        print('==' * 72)

    if save_path is not None:
        title = '\t'.join(['country', 'BCELoss', 'total_accuracy', 'hamming_accuracy', 'event_accuracy_sent',
                           'event_accuracy_label', 'mean_f1', 'mean_precision', 'mean_recall'])
        entries = []
        for country, metrics in results.items():
            metrics.pop("column_wise")
            entries.append('\t'.join([country, *list(map(str, metrics.values()))]))

        print(f'Saving results in {save_path + "/results.tsv"}')
        with open(os.path.join(save_path, "results.tsv"), 'w') as f:
            f.write(title)
            f.write('\n')
            f.write('\n'.join(entries))
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', help='name of the BERT model to load', type=str, default='xlm')
    parser.add_argument('-r', '--data_root', help='path to the directory containing test files per country', type=str, default=ROOT)
    parser.add_argument('-d', '--device', help='cpu or cuda', type=str, default='cuda')
    parser.add_argument('-len', '--max_length', help='truncate to maximum sentence length', type=int, default=256)
    parser.add_argument('-l', '--load_path', help='where to load pretrained model from', type=str)
    parser.add_argument('-s', '--save_path', help='where to save evaluation results (default no save)', type=str, default=None)
    parser.add_argument('-country', '--test_only_one', help='select a specific country to test (default all)', type=str, default=None)
    
    kwargs = vars(parser.parse_args())
    main(**kwargs)