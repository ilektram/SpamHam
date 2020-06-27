import random

from utils import read_data_as_df, preprocess_data, get_trained_models, evaluate_models, get_tfidf, get_feature_df, LOG
from config import DATA_PATH

random.seed(2)


if __name__ == "__main__":
    df = read_data_as_df(DATA_PATH)

    new_df = get_feature_df(df)
    tfidf_df = get_tfidf(new_df)

    X, y = preprocess_data(df)
    X_val, y_val = X.loc[X.index == 'VALIDATION'], y.loc[y.index == 'VALIDATION'].values
    X_test, y_test = X.loc[X.index == 'TEST'], y.loc[y.index == 'TEST'].values
    X_train, y_train = X.loc[X.index == 'TRAIN'], y.loc[y.index == 'TRAIN'].values
    LOG.info(f"Training set: {X_train.shape}, Testing set: {X_test.shape}, Validation set: {X_val.shape}")

    clf_d = get_trained_models(["RF", "SGD", "LR", "SVM"], X_train, y_train)
    evaluate_models(clf_d, X_train, X_val, y_train, y_val)
