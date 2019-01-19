from sklearn.model_selection import train_test_split

from utils import LOG, read_data_as_df, get_column_names, preprocess_data, get_trained_models, evaluate_models
from config import DATA_PATH, COL_NAME_PATH

if __name__ == "__main__":

    col_names_l = get_column_names(COL_NAME_PATH)

    df = read_data_as_df(DATA_PATH, names=col_names_l)
    LOG.debug(df.columns)

    LOG.debug(df.head(10))

    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=27, shuffle=True)

    clf_d = get_trained_models(["RF", "SGD", "LR", "SVM"], X_train, y_train)
    evaluate_models(clf_d, X_train, X_test, y_train, y_test)
