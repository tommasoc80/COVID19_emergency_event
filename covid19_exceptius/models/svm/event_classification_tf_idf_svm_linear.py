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

def cross_validation_linear_v1(df_text_train, df_labels_train):

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7)))
                              ])

    folds = KFold(n_splits=10, shuffle=True, random_state=42)

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
    ])

#    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,3), stop_words='english')
#    paragraph_tfidf = tfidf_vectorizer.fit_transform(df_text_train)

#    clf = OneVsRestClassifier(LinearSVC(), n_jobs=1)
#    clf = OneVsRestClassifier(SVC(kernel='linear'))

    # Train and test the model:
    scores = cross_val_score(clf,
                             df_text_train,
                             df_labels_train,
                             scoring='accuracy',
                             cv=folds)

    #print(scores)
    print("Accuracy one-hot-encoding 10-fold evaluation:" + str(scores.mean()))

    #cross_predictions = cross_val_predict(clf,
    #                                      paragraph_tfidf,
    #                                      df_labels_train,
    #                                      cv=10)

    #print(cross_predictions)


def random_train_test_split_v1(df_text_train, df_labels_train):

    X_train, X_test, y_train, y_test = train_test_split(df_text_train, df_labels_train, test_size = 0.20, random_state = 42)

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7)))
                               ])


    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
        ])

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)

    print("Accuracy one-hot-encoding - random train-test split: " + str(accuracy_score(y_test, prediction)))
    print("Hamming Loss one-hot-encoding - random train-test split: " + str(hamming_loss(y_test, prediction)))


def cross_validation_linear_v2(df_multilabel_v2):
    """

    :param df_multilabel_v2: dataframe with texts and classes (one per column)
    :return:
    """

    paragraph_tfidf_v2 = df_multilabel_v2["Paragraph text"]
    paragraph_labels_v2 = np.asarray(df_multilabel_v2[df_multilabel_v2.columns[1:]])

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1,3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3,7)))
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
    print("Accuracy 10-fold evaluation multilabel:" + str(scores.mean()))


def random_train_test_split_v2(df_multilabel_v2):
    """
    :param df_multilabel_v2: dataframe with texts and classes (one per column)
    :return:
    """

    paragraph_tfidf_v2 = df_multilabel_v2["Paragraph text"]
    paragraph_labels_v2 = np.asarray(df_multilabel_v2[df_multilabel_v2.columns[1:]])

    # splitting the data to training and testing data set
    X_train, X_test, y_train, y_test = train_test_split(paragraph_tfidf_v2, paragraph_labels_v2, test_size=0.20, random_state=42)

    vectorizer = FeatureUnion([('tfidf_word', TfidfVectorizer(ngram_range=(1, 3), stop_words='english')),
                               ('tfidf_char', TfidfVectorizer(analyzer='char', ngram_range=(3, 7)))
                               ])

    clf = Pipeline([
        ('vectorize', vectorizer),
        ('classify', OneVsRestClassifier(LinearSVC(), n_jobs=1))
        ])

    clf.fit(X_train, y_train)
    prediction = clf.predict(X_test)


    print("Accuracy - random train-test split: " + str(accuracy_score(y_test, prediction)))
    print("Hamming Loss - random train-test split: " + str(hamming_loss(y_test, prediction)))


if __name__ == '__main__':

    # multilabel classification - v1
    # the multilable classes are represented as one-hot-encoding vector

    data_in_v1 = 'event_classes_multilabel_v1.csv'
    data_class_v1 = pd.read_csv(data_in_v1, sep=',', header=0)
    class_distributions_v1 = data_class_v1['Classification_label'].value_counts()

    textual_data_v1 = data_class_v1['Paragraph text']
    labels_v1 = data_class_v1[['Classification_label']]

    cross_validation_linear_v1(textual_data_v1,labels_v1)
    #random_train_test_split_v1(textual_data_v1,labels_v1)

    # multilabel classification - v2
    # multilabel classes are represented separetly

    data_in_v2 = 'event_classes_multilabel_v2.csv'
    data_class_v2 = pd.read_csv(data_in_v2, sep=',', header=0)

    cross_validation_linear_v2(data_class_v2)
    #random_train_test_split_v2(data_class_v2)


