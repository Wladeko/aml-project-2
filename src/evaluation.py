import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier


def search_knn(X, y, num_folds=5, njobs=8, verbose=1):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_grid = {
        "n_neighbors": [1, 2, 3, 5, 9, 15, 19, 23, 27],
        "weights": ["uniform", "distance"],
        "p": [1, 2],
    }
    knn = KNeighborsClassifier()

    grid_search = GridSearchCV(knn, param_grid, scoring="balanced_accuracy", cv=skf, verbose=verbose, n_jobs=njobs)
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def search_rf(X, y, num_folds=5, njobs=8, verbose=1):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_grid = {"n_estimators": [50, 100, 200, 300, 400]}
    rf = RandomForestClassifier()

    grid_search = GridSearchCV(rf, param_grid, scoring="balanced_accuracy", cv=skf, verbose=verbose, n_jobs=njobs)
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def search_xgboost(X, y, num_folds=5, njobs=8, verbose=1):
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)
    param_grid = {"max_depth": [5, 7, 9, 11], "learning_rate": [0.01, 0.005], "n_estimators": [200, 300, 400]}
    # param_grid = {"max_depth": [7], "learning_rate": [0.01], "n_estimators": [300]}
    xgb = XGBClassifier()

    grid_search = GridSearchCV(xgb, param_grid, scoring="balanced_accuracy", cv=skf, verbose=verbose, n_jobs=njobs)
    grid_search.fit(X, y)

    return grid_search.best_params_, grid_search.best_score_


def evaluate(X, y, num_folds=5, scoring_coefficient=0.2, params_knn=None, params_rf=None, params_xgb=None):
    # scoring_coefficient should be 0.2 for artificial dataset and 0.01 for spam dataset
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    knn_scores = []
    rf_scores = []
    xgb_scores = []

    num_features = X.shape[1]

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # KNN Classifier
        knn = KNeighborsClassifier(**params_knn)
        knn.fit(X_train, y_train)
        knn_pred = knn.predict(X_test)
        knn_scores.append(balanced_accuracy_score(y_test, knn_pred))

        # Random Forest Classifier
        rf = RandomForestClassifier(**params_rf)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_scores.append(balanced_accuracy_score(y_test, rf_pred))

        # XGBoost Classifier
        xgb = XGBClassifier(**params_xgb)
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict(X_test)
        xgb_scores.append(balanced_accuracy_score(y_test, xgb_pred))

    avg_knn_score = np.mean(knn_scores)
    avg_rf_score = np.mean(rf_scores)
    avg_xgb_score = np.mean(xgb_scores)

    final_knn_score = avg_knn_score - 0.01 * (scoring_coefficient * num_features - 1)
    final_rf_score = avg_rf_score - 0.01 * (scoring_coefficient * num_features - 1)
    final_xgb_score = avg_xgb_score - 0.01 * (scoring_coefficient * num_features - 1)

    return {
        "KNN": (round(avg_knn_score, 4), round(final_knn_score, 4)),
        "RF": (round(avg_rf_score, 4), round(final_rf_score, 4)),
        "XGB": (round(avg_xgb_score, 4), round(final_xgb_score, 4)),
    }
