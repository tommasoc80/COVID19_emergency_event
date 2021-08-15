from covid19_exceptius.types import *

import os
import string
import pandas as pd
import numpy as np
import subprocess
from random import sample, seed
from tqdm import tqdm
from math import ceil
from collections import Counter
from itertools import zip_longest 

seed(42)


# some paragraphs start with: (1), (1)-, 1- etc.
# remove this part until the actual text begins 
def denoise_text(text: List[Sentence]) -> List[Sentence]:
    noise = [sent for sent in text if sent[0] not in string.ascii_letters]
    rest = [sent for sent in text if sent not in noise]
    noise = [' '.join(sent.strip().split(' ')[1:]) for sent in noise]
    return rest + noise

# 
# take original labels e.g.'(1,0,1,0,0,0,0,0)' and generate
# binary versions: 0->No Event, 1-> Some Event and
# main_event_versions: 0 -> No_Event, 1,...,5->Main_event, 6->Mixed events
def refine_labels(csv_path: str, write_path: str, num_main_events: int=5) -> None:
    # get old data
    text, labels = zip(*pd.read_csv(csv_path, sep=',', header=0).values.tolist())
    # count label distribution and sort in descending order
    labels_count = {k: v for k,v in sorted(Counter(labels).items(), key=lambda x: x[1], reverse=True)}

    # seperate main to secondary events based on label occurences
    main_events = {}
    for i, item in enumerate(labels_count.items()):
        main_events[item[0]] = i if i<=num_main_events else num_main_events+1

    # create a new file with generated labels
    with open(write_path, 'a+') as f:
        f.write("Paragraph text,Classification_label,Binary_label,Main_events_label\n")
        for sent, label in zip(text, labels):
            binary_label = 0 if label == '0,0,0,0,0,0,0,0' else 1
            main_label = main_events[label]
            f.write(','.join(['"'+sent+'"', '"'+label+'"', '"'+str(binary_label)+'"', '"'+str(main_label)+'"']))
            f.write('\n')


# merge annotations to a single tsv
def merge_tsvs(tsv_paths: List[str], append_path: str) -> None:
    text, labels = [], []
    # list all english translations from all csv, as well as their
    # 8 type event annotations
    for path in csv_paths:
        data = pd.read_csv(path, sep=',', header=0)
        text.extend([s.strip().replace('"',"`") for s in list(data['LEGAL TEXT_EN']) if type(s) == str])
        labels.extend([','.join(list(map(str, list(map(int, list(typevents)))))) for typevents in \
            zip(*[data['TYPEVENT' + str(i)] for i in range(1,9)])])


    # append to csv with original UK data
    with open(append_path, 'a+') as f:
        for sent, label in zip(text, labels):
            binary_label = 0 if label == '0,0,0,0,0,0,0,0' else 1 
            main_label = 0 #dummy main label event - will not use
            f.write(','.join(['"'+sent+'"', '"'+label+'"', '"'+str(binary_label)+'"', '"'+str(main_label)+'"']))
            f.write('\n')


def _str(s: Any):
    if type(s) == str:
        return s 
    else:
        return 'nan'


def nan_to_zero(seq: List[Any]) -> List[float]:
    return [1. if type(x) == str else 0. if np.isnan(x) else x for x in seq]


# parse subevent annotation to transfer to main type event
def parse_subevents(header: str, data: List[List[Any]]) -> List[List[float]]:
    sub_idces = [i for i, c in enumerate(header) if c.startswith('SUB')]
    num_events = len([c for c in header if c.startswith('TYPE')])
    pointers = {k: [] for k in range(1, 1 + num_events)}
    for i in sub_idces:
        pointer = int(header[i].split('SUBTYPEVENT')[1][0])
        pointers[pointer].append(i)
    sub_events = [[[d[x] for x in pointers[i]] for i in range(1, 1 + num_events)] for d in data]
    sub_events = [list(map(nan_to_zero, se)) for se in sub_events]
    sub_events = [[bool(sum(seq)) for seq in se] for se in sub_events]
    return sub_events


# parse data from original xlsx files (first saved as csv)
def parse_from_xlsx(in_file: str) -> Tuple[List[Any], ...]:
    with open(in_file, 'r+') as f:
        header = f.readlines()[0].strip('\n').split(',')
    data = pd.read_csv(in_file).values.tolist()

    # sort cols according to header, also to the data table 
    sorted_cols = sorted(range(len(header)), key=lambda k: header[k]) 
    header = [header[c] for c in sorted_cols]
    data = [[d[c] for c in sorted_cols] for d in data]
    
    sen_ids = [_str(d[header.index('SENID')]) for d in data]
    text_or = ([d[header.index('LEGAL_TEXT_OR')] for d in data])
    text_en = ([d[header.index('LEGAL TEXT_EN')] for d in data])
    
    col_events = [i for i, c in enumerate(header) if c.startswith('TYPEVENT')]
    events = [[bool(d[c]) for c in col_events] for i, d in enumerate(data)]

    # transfer subevents annotations to main events with logical or
    sub_events = parse_subevents(header, data)
    events = [[x or y for x, y in zip(e, se)] for e, se in zip(events, sub_events)]
    
    return sen_ids, text_or, text_en, events, header


# remove non string row enties and strip newlines entangled with text
def remove_junk(text: List[str]) -> Tuple[List[str], List[int]]:
    removed_idces = []; res = []
    for i, t in enumerate(text):
        if type(t) != str:
            removed_idces.append(i)
            continue
        else:
            # strip junk characters
            if '\t' in t:
                t = t.replace('\t', '')
            if '"' in t:
                t = t.replace('"', '')
            if '\n' in t:
                t = t.replace('\n', '')
            res.append(t)
    return res, removed_idces


# convert original csv files to desired tsv format
def convert_to_tsv(in_file: str, out_file_trans: str, out_file_raw: str) -> None:
    ids, text_or, text_en, events, header = parse_from_xlsx(in_file)
    text_en, removed_idces_en = remove_junk(text_en)
    text_or, removed_idces_or = remove_junk(text_or)
    ids_or = [_id for i, _id in enumerate(ids) if i not in removed_idces_or]
    ids_en = [_id for i, _id in enumerate(ids) if i not in removed_idces_en]
    events_en = [e for i, e in enumerate(events) if i not in removed_idces_en]
    events_or = [e for i, e in enumerate(events) if i not in removed_idces_or]
    assert len(text_or) == len(events_or) == len(ids_or)
    assert len(text_en) == len(events_en) == len(ids_en)
    
    strings_or = ['\t'.join([_id, t, *list(map(str, e))]) for _id, t, e in zip(ids_or, text_or, events_or)]
    strings_en = ['\t'.join([_id, t, *list(map(str, e))]) for _id, t, e in zip(ids_en, text_en, events_en)]

    # create translations tsv
    with open(out_file_trans, 'w+') as f:
        f.write('\t'.join(['id', 'text', *['event' + str(i) for i in range(1, 1 + len(events[0]))]]))
        f.write('\n')
        f.write('\n'.join(strings_en))

    # create original text tsv
    with open(out_file_raw, 'w+') as g:
        g.write('\t'.join(['id', 'text', *['event' + str(i) for i in range(1, 1 + len(events[0]))]]))
        g.write('\n')
        g.write('\n'.join(strings_or))


def write_to_tsv(out_file: str, data: List[AnnotatedSentence]):
    header = '\t'.join(['id', 'text', *['event' + str(i) for i in range(1,9)]])
    with open(out_file, 'w+') as f:
        f.write(header)
        f.write('\n')
        for i, sample in enumerate(data):
            text = sample.text.replace('"','').replace('\n','').replace('\t','').replace('\r','')
            #text = sample.text
            labels = sample.labels  
            assert len([l for l in labels if type(l) == bool]) == len(labels), (i, sample)
            write = '\t'.join([str(i), str(text), *list(map(str, labels))])
            assert len(write.split('\t')) == 2 + len(labels)
            f.write(write)
            f.write('\n') 


def read_labeled(file_path: str, num_labels: int = 8) -> List[AnnotatedSentence]:
    data = pd.read_table(file_path).values.tolist()
    return [AnnotatedSentence(col[0], col[1], col[2 : num_labels + 2]) for col in data]


def read_unlabeled(file_path: str) -> List[Sentence]:
    data = pd.read_table(file_path).values.tolist()
    return [Sentence(col[0], col[1]) for col in data]


def read_processed(file_path: str) -> List[Sentence]:
    with open(file_path, 'r') as f:
        lines = [l.split('\t')[0] for l in f]
    return [Sentence(no=i, text=l) for i, l in enumerate(lines)]


def read_tokenized(file_path: str) -> List[Sequence[int]]:
    with open(file_path, 'r') as f:
        tokens = [list(map(int, l.split())) for l in f]
    return tokens


def extract_class_weights(ds: List[AnnotatedSentence]) -> List[float]:
    labels_per_q = list(zip(*[s.labels for s in ds]))
    qs_neg = [[label for label in q if label == False] for q in labels_per_q]
    qs_pos = [[label for label in q if label == True] for q in labels_per_q]
    return [len(n) / len(p) for n, p in zip(qs_neg, qs_pos)]


def split_train_dev_test(ds: List[Any], sizes: Tuple[float, float, float] = (.8, .1, .1), dev_thresh: Maybe[int] = None, 
                         test_thresh: Maybe[int] = None) -> Tuple[List[Any], ...]:
    train_size, dev_size, test_size = sizes
    dev_thresh = len(ds) if dev_thresh is None else dev_thresh
    test_thresh = len(ds) if test_thresh is None else test_thresh
    dev_size = min(dev_thresh, ceil(dev_size * len(ds)))
    test_size = min(test_thresh, ceil(test_size * len(ds)))
    train_size = len(ds) - dev_size - test_size
    train = sample(ds, train_size)
    rest = [s for s in ds if s not in train]
    dev = sample(rest, dev_size)
    test = [s for s in rest if s not in dev]
    return train, dev, test


def split_documents_to_folders(root: str, sizes: Sequence[int], dev_thresh: int, test_thresh: int):

    def do_split(split: str, file: str, country: str):
        _filename =  file.split('/')[-1]
        _path = os.path.join(root, split, country)
        if not os.path.isdir(_path):
            os.mkdir(_path)
        subprocess.call(['mv', file, os.path.join(_path, _filename)])

    countries = [f for f in os.listdir(os.path.join(root, 'processed')) if os.path.isdir(os.path.join(root, 'processed', f))]
    all_files = [[os.path.join(root, 'processed', country, f) for f in  os.listdir(os.path.join(root, 'processed', country)) if f.endswith('txt')] 
                    for country in countries]

    for country, files in zip(countries, tqdm(all_files)):
        train, dev, test = split_train_dev_test(files, sizes, dev_thresh, test_thresh)
        
        for file in train:
            do_split('train', file, country)
        
        for file in dev:
            do_split('dev', file, country)
        
        for file in test:
            do_split('test', file, country)


# def concat_to_len(tokens: List[Sequence[int]], state: Maybe[List[int]] = None, max_len: int = 256) -> List[Sequence[int]]:
#     tokens = list(set([tuple(line) for line in tokens])) # convert to hashable type
#     to_merge = [line for line in tokens if len(line) <= max_len // 2]
#     if not to_merge or to_merge == state:
#         # recursion base case
#         return tokens

#     else:
#         group1 = sorted(to_merge[:len(to_merge) // 2], key=lambda l: len(l), reverse=True)
#         group2 = sorted(to_merge[len(to_merge) // 2:], key=lambda l: len(l), reverse=False)
#         grouped = [line1 + line2 for line1, line2 in zip_longest(group1, group2, fillvalue=tuple([]))]
#         rest = set(tokens).difference(set(to_merge))
#         rest.update(grouped)
#         return concat_to_len(list(rest), state=to_merge)
       

# def prepare_pretrain_corpus(root: str, tokenizer: Tokenizer, save_path: Maybe[str] = None) -> List[int]:
#     # load all text
#     countries = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
#     files = sum([[os.path.join(country, f) for f in  os.listdir(country) if f.endswith('txt')] for country in countries], [])
#     all_lines = []
#     print('Loading all txt files...')
#     for file in tqdm(files):
#         with open(file, 'r') as f:
#             all_lines.extend([l.split('\t')[0] for l in (line.strip() for line in f) if l])

#     # remove duplicates and very short lines
#     all_lines = list(set(all_lines))
#     all_lines = [l for l in all_lines if len(l.split()) >= 3]

#     # tokenize lines and shuffle
#     print('Tokenizing text...')
#     all_lines = list(map(tokenizer, all_lines))
#     all_lines = sample(all_lines, len(all_lines))

#     # concat sentences in single line with max len 256
#     all_lines = concat_to_len(all_lines)

#     # write to file if wanted
#     if save_path is not None:
#         print(f'Writing to {save_path + "/full.txt"}...')
#         to_strings = [' '.join(list(map(str, line))) for line in all_lines]
#         with open(save_path + '/full.txt', 'w') as f:
#             f.write('\n'.join(to_strings))

#     return all_lines


def concat_to_len(tokens: List[int], stop_token: int, 
                   start_token: int = 0, end_token: int = 2, max_len: int = 256) -> List[str]:
    # remove sos and eos tokens as they will be add manually
    tokens = tokens[1:-1]

    # find positions of all stop sequence tokens
    stop_pos = [i for i, t in enumerate(tokens) if t == stop_token]

    start, end, offset = -1, 1, -1
    grouped = []
    while 1:
        to_search = np.array(stop_pos[offset+1:]) - end + 1

        if len(to_search) == 0:
            break

        elif to_search[0] > max_len - 2:
            offset += 1 
            end = start + max_len - 2
            
        else:
            offset += 1 + np.where(to_search <= max_len - 2)[0][-1]
            end = stop_pos[offset]

        grouped.append([start_token] + tokens[start+1: end+1] + [end_token])
        start = end

    return grouped


def prepare_pretrain_corpus(root: str, tokenizer: Tokenizer, save_path: Maybe[str] = None) -> List[int]:
    # load all text
    countries = [os.path.join(root, f) for f in os.listdir(root) if os.path.isdir(os.path.join(root, f))]
    files = [[os.path.join(country, f) for f in  os.listdir(country) if f.endswith('txt')] for country in countries]
    all_lines = []
    for fs, country in zip(files, countries):
        print(f'Loading {country} files...')
        lines_now = []
        for file in tqdm(fs):
            with open(file, 'r') as f:
                lines_now.extend([l.split('\t')[0] for l in (line.strip() for line in f) if l])
        all_lines.append(lines_now)

    # tokenize and concat to default max len 256
    all_tokens = []
    stop = tokenizer.get_vocab()['.']
    print('Tokenizing...')
    for i, lines in enumerate(tqdm(all_lines)):
        _text = ' '.join(lines)
        _tokens = tokenizer.encode(_text)
        all_tokens.append(concat_to_len(_tokens, stop_token=stop))

    # write to file if wanted
    all_tokens = sum(all_tokens, [])
    if save_path is not None:
        print(f'Writing to {save_path + "/full.txt"}...')
        to_strings = [' '.join(list(map(str, line))) for line in all_tokens]
        with open(save_path + '/full.txt', 'w') as f:
            f.write('\n'.join(to_strings))

    return all_tokens