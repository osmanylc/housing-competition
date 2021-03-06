from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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


def create_one_hot_encoding(X, y, X_eval):
    # collection of columns w/ missing values
    cols_w_missing = get_cols_w_missing(X)

    # Create column with 1s if a value if missing
    X, X_eval = [create_is_missing_cols(A, cols_w_missing)
                 for A in [X, X_eval]]

    # Create one-hot encoding of categorical features
    X, X_eval = [pd.get_dummies(A) for A in [X, X_eval]]
   
    # Match features in training and test sets
    X, X_eval = X.align(X_eval, join='left', axis=1)

    return X, X_eval


def prepare_data(X, y, X_eval, test_size=.2):
    """
    Create clean training and validation sets.
    """
    X, X_eval = create_one_hot_encoding(X, y, X_eval)

    # Split data into test/validation sets
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=test_size,
                                          random_state=1337)

    return X, y, X_eval, X_t, X_v, y_t, y_v
