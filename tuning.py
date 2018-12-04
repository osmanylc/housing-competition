import numpy as np
from xgboost import XGBRegressor
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

seed = 1337

def pick_xgbr_hyperparams(X, y, n_est_range, lr_range, n):
    n_est_set = np.random.randint(*n_est_range, n)
    lr_set = np.random.uniform(*lr_range, n)
    
    cv_scores = []
    
    for n_est, lr in zip(n_est_set, lr_set):
        cv_scores.append(_get_xgbr_cv(X, y, n_est, lr))
        print(cv_scores[-1])
    
    return min(cv_scores, key=lambda x: x['cv_error'])


def _get_xgbr_cv(X, y, n_est, lr):
    xgbr = make_pipeline(SimpleImputer(), 
                         XGBRegressor(n_estimators=n_est, learning_rate=lr, 
                                      random_state = seed))
    cv = cross_val_score(xgbr, X, y, cv=5, scoring='neg_mean_absolute_error')

    return {'cv_error': -np.mean(cv), 'n_est': n_est, 'lr': lr}


def auto_search_xgbr_params(estimator, X, y, n_est_range, 
                            lr_range, rounds, is_log_scale):
    # for r in range(rounds):
    pass

