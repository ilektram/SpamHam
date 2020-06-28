import random

from utils import read_data_as_df, preprocess_data, get_trained_models, evaluate_models, get_tfidf, get_feature_df, LOG
from config import DATA_PATH

random.seed(2)


def run_training():
    df = read_data_as_df(DATA_PATH)

    new_df = get_feature_df(df)
    tfidf_df = get_tfidf(new_df)

    X, y = preprocess_data(tfidf_df)

    X_test, y_test = X.loc[X.index == 'TEST'], y.loc[y.index == 'TEST'].values
    X_train, y_train = X.loc[(X.index == 'TRAIN') | (X.index == 'VALIDATION')], y.loc[(y.index == 'TRAIN') | (y.index == 'VALIDATION')].values
    LOG.info(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    LOG.info(f"Training set positive examples: {y_train.sum()}, Testing set positive examples: {y_test.sum()}")

    clf_d = get_trained_models(["RF", "SGD", "LR", "SVM"], X_train, y_train)
    evaluate_models(clf_d, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    run_training()
