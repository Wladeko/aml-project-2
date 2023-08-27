import pandas as pd


def prepare_data(test=False):
    if not test:
        X = pd.read_csv("data/artificial_train.data", sep=" ", header=None).iloc[:, :-1]
        y = pd.read_csv("data/artificial_train.labels", header=None)

        X = X.to_numpy()
        y = y.to_numpy().ravel()

        # convert labels to 0 and 1
        y[y == -1] = 0

        return X, y

    else:
        X = pd.read_csv("data/artificial_valid.data", sep=" ", header=None).iloc[:, :-1]
        X = X.to_numpy()

        return X
