from covid19_exceptius.types import *

import string
import pandas as pd
import numpy as np
from collections import Counter 


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
def parse_subevents(header: str, data: List[List[Any]]) -> List[List[List[Any]]]:
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
            text = sample.text.replace('"','')
            labels = sample.labels  
            assert len([l for l in labels if type(l) == bool]) == len(labels), (i, sample)
            write = '\t'.join([_str(i), _str(text), *list(map(_str, labels))])
            assert len(write.split('\t')) == 2 + len(labels)
            f.write(write)
            f.write('\n') 


def read_labeled(file_path: str, num_labels: int = 8) -> List[AnnotatedSentence]:
    data = pd.read_table(file_path).values.tolist()
    return [AnnotatedSentence(col[0], col[1], col[2 : num_labels + 2]) for col in data]


def read_unlabeled(file_path: str) -> List[Sentence]:
    data = pd.read_table(file_path).values.tolist()
    return [Sentence(col[0], col[1]) for col in data]