import logging
import sys
import re
import string
from os import path

import joblib
from itertools import product
from collections import Counter
from typing import List, Dict
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer, pos_tag, word_tokenize
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve
from joblib import dump, load


from config import LOG_LEVEL, RF_RAND_GRID, LR_RAND_GRID, SGD_RAND_GRID, SVM_RAND_GRID, DATA_PATH, OUT_PATH

LOG = logging.getLogger("HamSpam")
levels = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG
}
LOG.setLevel(levels[LOG_LEVEL])

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(levels[LOG_LEVEL])
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
LOG.addHandler(ch)

stop_words = set(stopwords.words('english'))
lmtzr = WordNetLemmatizer()
tfidf_v = TfidfVectorizer(max_features=300, max_df=0.95)


def read_data_as_df(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, header=None, names=['Dataset', 'Text', 'SpamHam'])
    df['is_spam'] = np.where(df['SpamHam'].str.lower().str.strip() == 'spam', 1, 0)
    df.drop(['SpamHam'], inplace=True, axis=1)
    LOG.info("Missing values per column: {}".format(df.isna().sum()))
    LOG.info("Dropping rows with any missing values...")
    LOG.info("We have {} spam and {} ham messages.".format(df['is_spam'].sum(), len(df) - df['is_spam'].sum()))
    df.dropna(inplace=True, how='any')
    for col in df.columns:
        LOG.info(f"Some information about our data in column {col}: {df[col].describe()}")
    return df


def count_emojis(text: str) -> int:
    return len(re.findall(r'[^\w\s,]', text))


def count_pos_tags(text: str) -> Counter:
    return Counter(x[1] for x in pos_tag(text))


def count_punct(filtered_sentence: List[str]) -> int:
    return len([x for x in filtered_sentence if x in string.punctuation])


def clean_text(text: str, lemmatizer: WordNetLemmatizer = lmtzr) -> List:
    word_tokens = word_tokenize(text)
    filtered_sentence = [w.strip().lower() for w in word_tokens if w not in stop_words]
    n_tokens = len(word_tokens)
    n_stopwords = n_tokens - len(filtered_sentence)
    n_pos = count_pos_tags(text)
    n_punct = count_punct(filtered_sentence)
    n_urls = len(re.findall(r'(https?://[^\s]+)', text))
    lemmas = " ".join([lemmatizer.lemmatize(token) for token in filtered_sentence])
    return [lemmas, n_stopwords, n_punct, n_pos, n_urls, n_tokens]


def preprocess_data(df):
    df.set_index('Dataset', drop=True, append=False, inplace=True)
    y = df["is_spam"]
    X = df.drop(['Text', 'is_spam', 'feature_list', 'lemmas', 'n_pos'], axis=1)
    X.fillna(0, inplace=True)
    X[:1000].to_csv('test.csv')
    return X, y


def do_crossvalidation(model_name, features, labels, save=False):
    if model_name == "RF":
        classifier = RandomForestClassifier(min_samples_split=3, min_samples_leaf=3, class_weight='balanced',
                                            verbose=0, n_jobs=-1, random_state=7)
        param_grid = RF_RAND_GRID
    elif model_name == "LR":
        classifier = LogisticRegression(verbose=0, n_jobs=-1, random_state=7, max_iter=1000, class_weight='balanced')
        param_grid = LR_RAND_GRID
    elif model_name == "SGD":
        classifier = SGDClassifier(verbose=0, n_jobs=-1, random_state=7, max_iter=1000, class_weight='balanced',
                                   n_iter_no_change=10, early_stopping=True, loss='log')
        param_grid = SGD_RAND_GRID
    elif model_name == "SVM":
        classifier = SVC(verbose=0, random_state=7, max_iter=1000, probability=True, class_weight='balanced')
        param_grid = SVM_RAND_GRID
    else:
        LOG.error(f"Model {model_name} unrecognized. Please specify one of RF, LR, SGD, SVM!")
        return None
    classifier_random = RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=100,
        cv=5,
        verbose=2,
        random_state=72,
        n_jobs=-1,
        scoring=['accuracy', 'neg_log_loss', 'precision', 'recall'],
        refit='neg_log_loss',
        return_train_score=True
    )
    LOG.info(f"Training {model_name} model...")
    trained_model = classifier_random.fit(features, labels)
    results = trained_model.cv_results_
    best_model_idx = trained_model.best_index_

    LOG.info("Best params for {} model: {}".format(model_name, trained_model.best_params_))
    LOG.info("Best {} model: {} with negative log loss score: {}".format(model_name, trained_model.best_estimator_,
                                                                         trained_model.best_score_))
    for score_type in ['accuracy', 'neg_log_loss', 'precision', 'recall']:
        LOG.info("Top {} scorer {} mean score, std: training = {} ± {}, testing = {} ± {}\n".format(
            model_name,
            score_type,
            round(results["mean_train_{}".format(score_type)][best_model_idx], 3),
            round(results["std_train_{}".format(score_type)][best_model_idx], 3),
            round(results["mean_test_{}".format(score_type)][best_model_idx], 3),
            round(results["std_test_{}".format(score_type)][best_model_idx]), 3)
        )
    LOG.info("Mean fit time: {} with std {}".format(results["mean_fit_time"][best_model_idx],
                                                    results["std_fit_time"][best_model_idx]))

    if save:
        LOG.info(f"Saving classifier as {model_name}.joblib...")
        dump(trained_model, path.join(OUT_PATH, f'{model_name}.joblib'))
    return trained_model


def get_trained_models(model_names_l, features, labels):
    clf_d = {}
    for clf in model_names_l:
        if path.exists(path.join(OUT_PATH, f"{clf}.joblib")):
            trained_clf = load(path.join(OUT_PATH, f"{clf}.joblib"))
        else:
            trained_clf = do_crossvalidation(clf, features, labels, save=True)
        clf_d[clf] = trained_clf
    return clf_d


def plot_confusion_matrix(confusion, title, subtitle="", classes=("HAM", "SPAM"), show=False):
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Greens)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.title(title)
    plt.suptitle(subtitle)

    thresh = confusion.max() / 2.
    for i, j in product(range(confusion.shape[0]), range(confusion.shape[1])):
        plt.text(j, i, format(confusion[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if confusion[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if show:
        plt.show()
    return plt


def plot_roc_auc(fpr, tpr, auc, title, subtitle="", show=False):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='red',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.suptitle(subtitle)
    plt.legend(loc="lower right")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if show:
        plt.show()
    return plt


def evaluate_models(clf_d, X_train, X_test, y_train, y_test):
    for k, v in clf_d.items():
        y_train_pred = v.predict(X_train)
        y_train_pred_proba = v.predict_proba(X_train)
        y_test_pred = v.predict(X_test)
        y_test_pred_proba = v.predict_proba(X_test)
        LOG.info(f"Evaluating model {k}...")
        training_acc, training_loss = round(accuracy_score(y_train, y_train_pred), 3), round(
            log_loss(y_train, y_train_pred_proba), 3)
        testing_acc, testing_loss = round(accuracy_score(y_test, y_test_pred), 3), round(
            log_loss(y_test, y_test_pred_proba), 3)
        LOG.info("Training set accuracy:  {}, log loss: {}".format(training_acc, training_loss))
        LOG.info("Testing set accuracy:  {}, log loss: {}".format(testing_acc, testing_loss))

        confusion = confusion_matrix(y_test, y_test_pred)
        cm_p = plot_confusion_matrix(confusion, title="{} Confusion Matrix on Test Set".format(k),
                                     subtitle=f"Accuracy: {testing_acc}, Logloss: {testing_loss}")
        cm_p.savefig(path.join(OUT_PATH, f"ConfusionMatrix{k}.png"))
        cm_p.clf()

        fpr, tpr, thresholds = roc_curve(y_test, [n[1] for n in y_test_pred_proba])
        roc_auc = roc_auc_score(y_test, [n[1] for n in y_test_pred_proba])
        LOG.info(f"AUC score for {k} on test set {round(roc_auc, 3)}\n")

        roc_p = plot_roc_auc(fpr, tpr, roc_auc, title=f"ROC on Test Set for {k}",
                             subtitle=f"Accuracy: {testing_acc}, Logloss: {testing_loss}")
        roc_p.savefig(path.join(OUT_PATH, f"RocAuc{k}.png"))
        roc_p.clf()


def get_feature_df(df: pd.DataFrame) -> pd.DataFrame:
    LOG.info("Cleaning up the text...")
    df['feature_list'] = df['Text'].apply(clean_text)
    LOG.info("Extracted new features...")
    df[['lemmas', 'n_stopwords', 'n_punct', 'n_pos', 'n_urls', 'n_tokens']] = pd.DataFrame(df['feature_list'].tolist(), index=df.index)
    POS_df = pd.DataFrame(df['n_pos'].to_list(), index=df.index).fillna(0)
    LOG.debug("Got POS_df")
    final_df = df.combine_first(POS_df)
    LOG.debug("Got final_df")
    return final_df


def get_tfidf(df: pd.DataFrame, tfidf_vect: TfidfVectorizer = tfidf_v, fit: bool = True):
    df.dropna(subset=['lemmas'], inplace=True)
    LOG.debug(f"feature columns: {df.columns}")
    if fit:
        tfidf = tfidf_vect.fit_transform(df['lemmas'])
        joblib.dump(tfidf_vect, path.join(OUT_PATH, 'tfidf_vect.joblib'))
    else:
        tfidf = tfidf_vect.transform(df['lemmas'])
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=tfidf_vect.get_feature_names(), index=df.index)
    final_df = df.combine_first(tfidf_df)
    return final_df


if __name__ == "__main__":
    df = read_data_as_df(DATA_PATH)[:100]

    new_df = get_feature_df(df)
    tfidf_df = get_tfidf(new_df)

    X, y = preprocess_data(df)
    X_val, y_val = X.loc[X.index == 'VALIDATION'], y.loc[y.index == 'VALIDATION'].values
    X_test, y_test = X.loc[X.index == 'TEST'], y.loc[y.index == 'TEST'].values
    X_train, y_train = X.loc[X.index == 'TRAIN'], y.loc[y.index == 'TRAIN'].values
    LOG.info(f"{[len(X_val), len(X_train), len(y_val)]}")
    #
    # trained_model = load("RF.joblib")
    # results = trained_model.cv_results_
    # best_model_idx = trained_model.best_index_
    # model_name = "RF"
    #
    # LOG.info("Best params for {} model: {}".format(model_name, trained_model.best_params_))
    # LOG.info("Best {} model: {} with negative log loss score: {}".format(model_name, trained_model.best_estimator_,
    #                                                                      trained_model.best_score_))
