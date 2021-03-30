from covid19_exceptius.types import *

import string
import pandas as pd
from collections import Counter 


# some paragraphs start with: (1), (1)-, 1- etc.
# remove this part until the actual text begins 
def denoise_text(text: List[Sentence]) -> List[Sentence]:
    noise = [sent for sent in text if sent[0] not in string.ascii_letters]
    rest = [sent for sent in text if sent not in noise]
    noise = [' '.join(sent.strip().split(' ')[1:]) for sent in noise]
    return rest + noise


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


# merge English translation annotations to a single csv
def merge_english_csv(csv_paths: List[str], append_path: str) -> None:
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
    return '"' + str(s) + '"'


# convert original csv files to desired tsv format
def convert_to_tsv(in_file: str, out_file_trans: str, out_file_raw: str) -> None:
    header = '\t'.join(['id', 'text', *['event' + str(i) for i in range(1,9)]])
    data = pd.read_csv(in_file)

    valid_ids_trans = [i for i, s in enumerate(list(data['LEGAL TEXT_EN'])) if type(s) == str]
    valid_ids_raw = [i for i, s in enumerate(list(data['LEGAL_TEXT_OR'])) if type(s) == str]
    valid_ids = set(valid_ids_trans).intersection(set(valid_ids_raw))

    text_trans = [s for i, s in enumerate(list(data['LEGAL TEXT_EN'])) if i in valid_ids]
    text_raw = [s for i, s in enumerate(list(data['LEGAL_TEXT_OR'])) if i in valid_ids]
    labels = [[q for i, q in enumerate(list(data['TYPEVENT' + str(j)])) if i in valid_ids] for j in range(1,9)]

    # create translations tsv
    with open(out_file_trans, 'a+') as f:
        f.write(header)
        f.write('\n')
        for _id, (txt, l1, l2, l3, l4, l5, l6, l7, l8) in enumerate(zip(text_trans, *labels)):
            write = '\t'.join([_str(_id), _str(txt), _str(l1), _str(l2), _str(l3), _str(l4), \
                _str(l5), _str(l6), _str(l7), _str(l8)])
            f.write(write)
            f.write('\n')

    # create original text tsv
    with open(out_file_raw, 'a+') as g:
        g.write(header)
        g.write('\n')
        for _id, (txt, l1, l2, l3, l4, l5, l6, l7, l8) in enumerate(zip(text_raw, *labels)):
            write = '\t'.join([_str(_id), _str(txt), _str(l1), _str(l2), _str(l3), _str(l4), \
                _str(l5), _str(l6), _str(l7), _str(l8)])
            g.write(write)
            g.write('\n')


def read_labeled(file_path: str) -> List[AnnotatedSentence]:
    data = pd.read_table(file_path).values.tolist()
    return [AnnotatedSentence(col[0], col[1], col[2:]) for col in data]


def read_unlabeled(file_path: str) -> List[Sentence]:
    data = pd.read_table(file_path).values.tolist()
    return [Sentence(col[0], col[1]) for col in data]