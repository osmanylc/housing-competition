from sklearn.impute import SimpleImputer
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

