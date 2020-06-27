from os import getenv
import numpy as np

LOG_LEVEL = getenv('LOG_LEVEL', "INFO")
DATA_PATH = getenv("DATA_PATH", "~/Downloads/2020_Peanut_exercice_dataset.csv")

RF_RAND_GRID = {
    "criterion": ["gini", "entropy"],
    "max_features": ['auto', 'sqrt', "log2"],
    "n_estimators": [int(x) for x in np.linspace(start=10, stop=1000, num=10)],
    "max_depth": [int(x) for x in np.linspace(start=1, stop=5, num=1)]
}

LR_RAND_GRID = {'C': [0.01, 0.05, 0.1, 0.5, 1, 5, 10], 'penalty': ['l1', 'l2']}

SGD_RAND_GRID = grid = {
    'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
    'penalty': ['l2', "l1", "elasticnet"]
}

SVM_RAND_GRID = {
    "kernel": ["linear", "rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10],
    "gamma": [0.001, 0.01, 0.1, 1]
}
