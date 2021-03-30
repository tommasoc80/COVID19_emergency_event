import string
import pandas as pd
from collections import Counter 

from typing import List 

Sentence = List[str]
Sentences = List[Sentence]

# some paragraphs start with: (1), (1)-, 1- etc.
# remove this part until the actual text begins 
def denoise_text(text: Sentences) -> Sentences:
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
