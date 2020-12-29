import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score,cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.pipeline import Pipeline, FeatureUnion
import gensim.downloader as api # import the downloader and allow us to download and use them
from nltk import word_tokenize
import itertools
import embedding_vectorizer

def cross_validation_linear_v1(df_text_train, df_labels_train, embedding_vocabulary):

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7))),
                               ('embed', embedding_vectorizer.Embeddings(embedding_vocabulary))
                               ])

    folds = KFold(n_splits=10, shuffle=True, random_state=42)

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
    ])

    # Train and test the model:
    scores = cross_val_score(clf,
                             df_text_train,
                             df_labels_train,
                             scoring='accuracy',
                             cv=folds)

    #print(scores)
    print("Accuracy one-hot-encoding 10-fold evaluation:" + str(scores.mean()))

def random_train_test_split_v1(df_text_train, df_labels_train, embedding_vocabulary):

    X_train, X_test, y_train, y_test = train_test_split(df_text_train, df_labels_train, test_size = 0.20, random_state = 42)

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7))),
                               ('embed', embedding_vectorizer.Embeddings(embedding_vocabulary))
                               ])

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
        ])

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print("Accuracy one-hot-encoding - random train-test split: " + str(accuracy_score(y_test, prediction)))
    print("Hamming Loss one-hot-encoding - random train-test split: " + str(hamming_loss(y_test, prediction)))


def cross_validation_linear_v2(df_multilabel_v2, embedding_vocabulary):
    """

    :param df_multilabel_v2: dataframe with texts and classes (one per column)
    :return:
    """

    paragraph_tfidf_v2 = df_multilabel_v2["Paragraph text"]
    paragraph_labels_v2 = np.asarray(df_multilabel_v2[df_multilabel_v2.columns[1:]])

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7))),
                               ('embed', embedding_vectorizer.Embeddings(embedding_vocabulary))
                               ])

    folds = KFold(n_splits = 10, shuffle = True, random_state = 42)

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
    ])


#    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
#    clf = OneVsRestClassifier(SVC(kernel='linear'))

    # Train and test the model:
    scores = cross_val_score(clf,
                             paragraph_tfidf_v2,
                             paragraph_labels_v2,
                             scoring='accuracy',
                             cv=folds)

#    print(scores)
    print("Accuracy multilabel 10-fold evaluation:" + str(scores.mean()))


def random_train_test_split_v2(df_multilabel_v2, embedding_vocabulary):
    """
    :param df_multilabel_v2: dataframe with texts and classes (one per column)
    :return:
    """

    paragraph_tfidf_v2 = df_multilabel_v2["Paragraph text"]
    paragraph_labels_v2 = np.asarray(df_multilabel_v2[df_multilabel_v2.columns[1:]])

    # splitting the data to training and testing data set
    X_train, X_test, y_train, y_test = train_test_split(paragraph_tfidf_v2, paragraph_labels_v2, test_size=0.20, random_state=42)

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7))),
                               ('embed', embedding_vectorizer.Embeddings(embedding_vocabulary))
                               ])

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
        ])

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)


    print("Accuracy - random train-test split: " + str(accuracy_score(y_test, prediction)))
    print("Hamming Loss - random train-test split: " + str(hamming_loss(y_test, prediction)))

def load_embeddings(textual_data_train, embedding_vector):

    vocabulary_train = set(token.lower() for song in textual_data_train for token in song.split())

    embedding_lookup = {}

    for elem in vocabulary_train:
        # for i, elem in enumerate(itertools.islice(vocabulary_train, 10)):
        #    print(elem)
        if elem in embedding_vector:
            embedding_lookup[elem] = np.array(embedding_vector[elem])

    return embedding_lookup

if __name__ == '__main__':

    # multilabel classification - v1
    # the multilable classes are represented as one-hot-encoding vector

    print("Reading data for one-hot-encoding class...")
    data_in_v1 = 'event_classes_multilabel_v1.csv'
    data_class_v1 = pd.read_csv(data_in_v1, sep=',', header=0)
    class_distributions_v1 = data_class_v1['Classification_label'].value_counts()

    textual_data_v1 = data_class_v1['Paragraph text']
    labels_v1 = data_class_v1[['Classification_label']]

    print("Loading embeddings ...")

    embedding_vector = api.load("glove-wiki-gigaword-300")

    print("Creating embedding voc ...")

    embedding_vocabulary = load_embeddings(textual_data_v1,embedding_vector)

    print("Run cross-fold validation one-hot-encoding ...")

    cross_validation_linear_v1(textual_data_v1,labels_v1,embedding_vocabulary)
    #random_train_test_split_v1(textual_data_v1,labels_v1,embedding_vocabulary)

    # multilabel classification - v2
    # multilabel classes are represented separetly
    print("Reading data for multilabel ...")

    data_in_v2 = 'event_classes_multilabel_v2.csv'
    data_class_v2 = pd.read_csv(data_in_v2, sep=',', header=0)

    print("Run cross-fold validation multilabel ...")

    cross_validation_linear_v2(data_class_v2,embedding_vocabulary)
    #random_train_test_split_v2(data_class_v2,embedding_vocabulary)


