import pandas as pd

text, labels = zip(*pd.read_csv('Models/event_classes_multilabel_v1.csv', sep=',', header=0).values.tolist())