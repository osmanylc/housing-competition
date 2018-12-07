import numpy as np
import copy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

seed = 1337

def make_param_set(low, high, n, is_log=False, is_int=False):
    if is_log:
        low, high = [np.log(x) for x in [low, high]]
    
    step_size = (high - low) / (n-1)
    param_set = [low + step_size * i for i in range(n)]
    
    if is_log:
        param_set = np.exp(param_set)
        
    if is_int:
        param_set = [int(p) for p in param_set]
        
    return param_set


def make_all_param_sets(param_ranges):
    param_sets = {}
    for args in param_ranges:
        param_set = make_param_set(args['low'], args['high'], args['n'], args['is_log'], args['is_int'])
        param_sets[args['name']] = param_set
    
    return param_sets


def reduce_param_range(low, high, param, alpha, is_int=True, is_log=False):
    if is_log:
        low, high, param = np.log([low, high, param])
    
    param_range = high - low
    low, high = (max(low, param - alpha * param_range / 2), 
                 min(high, param + alpha * param_range / 2))
    
    if is_log:
        low, high = np.exp([low, high])
    
    if is_int:
        low, high = [int(x) for x in [low, high]]
        
    return low, high


def reduce_all_param_ranges(param_results, param_ranges, alpha):
    param_ranges = copy.deepcopy(param_ranges)
    
    for args in param_ranges:
        args['low'], args['high'] = reduce_param_range(args['low'], args['high'], 
                                                       param_results[args['name']], alpha, 
                                                       args['is_int'], args['is_log'])
    
    return param_ranges


def find_hyperparams_iterated(model, X, y, param_ranges, fit_params, n_samplings, scoring, cv_folds, n_iters, alpha):
    """
    Find best hyperparams based on cross-validation scores.
    
    param_ranges: list of param arguments to build a param set.
    """
    for i in range(n_iters):
        param_sets = make_all_param_sets(param_ranges)
        print(param_sets)
        rand_search_results = (RandomizedSearchCV(model, param_distributions=param_sets,
                                                 n_iter=n_samplings, scoring=scoring, 
                                                 cv=cv_folds, n_jobs=-1, random_state=seed)
                               .fit(X,y, **fit_params))
        best_params = rand_search_results.best_params_
        param_ranges = reduce_all_param_ranges(best_params, param_ranges, alpha)
    
    return best_params
