import logging
import sys
from os import path
from math import log
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, roc_auc_score, roc_curve
from joblib import dump, load

import numpy as np

from config import LOG_LEVEL, RF_RAND_GRID, LR_RAND_GRID, SGD_RAND_GRID, SVM_RAND_GRID


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


def get_column_names(filename):
    with open(filename, 'r') as f:
        content = f.readlines()
    col_names = [x.split()[0][:-1] for x in content[33:]]
    col_names.append("target")
    return col_names


def read_data_as_df(filename, names=None):
    LOG.info(f"Reading data from file {filename}...")
    if not names:
        df = pd.read_csv(filename, header=None)
    else:
        df = pd.read_csv(filename, names=names)
    LOG.debug(f"Missing values per column: {df.isna().sum()}")
    LOG.info("Dropping rows with any missing values...")
    df.dropna(inplace=True, how='any')
    for col in df.columns:
        LOG.info(f"Some information about our data in column {col}: {df[col].describe()}")
    return df


def preprocess_data(df):
    scaler = MinMaxScaler()
    y = df["target"].values
    X = df.drop("target", axis=1)
    for c in X.columns:
        try:
            X[c + "_log"] = X[c].apply(lambda z: log(z))
        except ValueError:
            continue
    scaled_X = scaler.fit_transform(X)
    return scaled_X, y


def do_crossvalidation(model_name, features, labels, save=False):
    if model_name == "RF":
        classifier = RandomForestClassifier(min_samples_split=3, min_samples_leaf=3, class_weight='balanced',
                                            verbose=0, n_jobs=-1, random_state=7)
        param_grid = RF_RAND_GRID
    elif model_name == "LR":
        classifier = LogisticRegression(verbose=0, n_jobs=-1, random_state=7, class_weight='balanced', max_iter=1000)
        param_grid = LR_RAND_GRID
    elif model_name == "SGD":
        classifier = SGDClassifier(verbose=0, n_jobs=-1, random_state=7, class_weight='balanced', n_iter=1000,
                                   n_iter_no_change=10, early_stopping=True, loss='log')
        param_grid = SGD_RAND_GRID
    elif model_name == "SVM":
        classifier = SVC(verbose=0, random_state=7, class_weight='balanced', max_iter=1000, probability=True)
        param_grid = SVM_RAND_GRID
    else:
        LOG.ERROR(f"Model {model_name} unrecognized. Please specify one of RF, LR, SGD, SVM!")
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
        refit='neg_log_loss'
    )
    LOG.info(f"Training {model_name} model...")
    trained_model = classifier_random.fit(features, labels)
    results = trained_model.cv_results_
    best_model_idx = trained_model.best_index_

    LOG.info("Best params for {} model: {}".format(model_name, trained_model.best_params_))
    LOG.info("Best {} model: {} with negative log loss score: {}".format(model_name, trained_model.best_estimator_, trained_model.best_score_))
    for score_type in ['accuracy', 'neg_log_loss', 'precision', 'recall']:
        LOG.info("Top {} scorer {} mean score, std: training = {} ± {}, testing = {} ± {}\n".format(
            model_name,
            score_type,
            round(results["mean_train_{}".format(score_type)][best_model_idx], 3),
            round(results["std_train_{}".format(score_type)][best_model_idx], 3),
            round(results["mean_test_{}".format(score_type)][best_model_idx], 3),
            round(results["std_test_{}".format(score_type)][best_model_idx]), 3)
        )
    LOG.info("Mean fit time: {} with std {}".format(results["mean_fit_time"][best_model_idx], results["std_fit_time"][best_model_idx]))

    if save:
        LOG.info(f"Saving classifier as {model_name}.joblib...")
        dump(trained_model, f'{model_name}.joblib')
    return trained_model


def get_trained_models(model_names_l, features, labels):
    clf_d = {}
    for clf in model_names_l:
        if path.exists(f"{clf}.joblib"):
            trained_clf = load(f"{clf}.joblib")
        else:
            trained_clf = do_crossvalidation(model_name, features, labels, save=True)
        clf_d[clf] = trained_clf
    return clf_d


def plot_confusion_matrix(confusion, title, subtitle="", classes=["HAM", "SPAM"], show=False):
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
        training_acc, training_loss = round(accuracy_score(y_train, y_train_pred), 3), round(log_loss(y_train, y_train_pred_proba), 3)
        testing_acc, testing_loss = round(accuracy_score(y_test, y_test_pred), 3), round(log_loss(y_test, y_test_pred_proba), 3)
        LOG.info("Training set accuracy:  {}, log loss: {}".format(training_acc, training_loss))
        LOG.info("Testing set accuracy:  {}, log loss: {}".format(testing_acc, testing_loss))

        confusion = confusion_matrix(y_test, y_test_pred)
        cm_p = plot_confusion_matrix(confusion, title="{} Confusion Matrix on Test Set".format(k),
                                  subtitle=f"Accuracy: {testing_acc}, Logloss: {testing_loss}")
        cm_p.savefig(f"ConfusionMatrix{k}.png")
        cm_p.clf()

        fpr, tpr, thresholds = roc_curve(y_test, [n[1] for n in y_test_pred_proba])
        roc_auc = roc_auc_score(y_test, [n[1] for n in y_test_pred_proba])
        LOG.info(f"AUC score for {k} on test set {round(roc_auc, 3)}\n")

        roc_p = plot_roc_auc(fpr, tpr, roc_auc, title=f"ROC on Test Set for {k}", subtitle=f"Accuracy: {testing_acc}, Logloss: {testing_loss}")
        roc_p.savefig(f"RocAuc{k}.png")
        roc_p.clf()


if __name__ == "__main__":
    col_names_l = get_column_names("spambase/spambase.names")

    df = read_data_as_df("spambase/spambase.data", names=col_names_l)
    LOG.debug(df.columns)

    LOG.debug(df.head(10))

    X, y = preprocess_data(df)

    trained_model = load("RF.joblib")
    results = trained_model.cv_results_
    best_model_idx = trained_model.best_index_
    model_name = "RF"

    LOG.info("Best params for {} model: {}".format(model_name, trained_model.best_params_))
    LOG.info("Best {} model: {} with negative log loss score: {}".format(model_name, trained_model.best_estimator_,
                                                                         trained_model.best_score_))


