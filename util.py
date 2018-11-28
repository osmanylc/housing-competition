from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error

import pandas as pd


def get_cols_w_missing(X):
    return X.columns[X.isna().any()]


def create_is_missing_cols(X, cols_w_missing):
    X = X.copy()

    for col in cols_w_missing:
        X[col + 'is_missing'] = X[col].isna()

    return X


def impute_missing(X_train, X_test):
    features = X_train.columns
    imp = SimpleImputer().fit(X_train)
    
    X_train, X_test = (imp.transform(X_train), 
                       imp.transform(X_test))
    X_train, X_test = (pd.DataFrame(X_train, columns=features),
                       pd.DataFrame(X_test, columns=features))

    return X_train, X_test


def get_mae(X_train, y_train, X_test, y_test, mln):
    rf = RandomForestRegressor(n_estimators=500, criterion='mse', 
                               max_leaf_nodes=mln, random_state=1337)
    rf.fit(X_train, y_train)
    y_hat = rf.predict(X_test)
    
    return mean_absolute_error(y_test, y_hat)


