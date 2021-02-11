# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:05:44 2020

@author: Alok Garg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import xgboost as xgb


def grid_search(parameters,x,y):
    classifier = xgb.XGBRFClassifier(objective="multi:softprob", verbosity=2,n_jobs = -1)
    gs = GridSearchCV(estimator = classifier,param_grid = parameters, scoring= None,n_jobs=-1 ,verbose=3,cv=3)
    # GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)[source]¶
    gs.fit(x,y)
    print(gs.best_params_)
    result_df = pd.DataFrame(gs.cv_results_)
    return result_df










if __name__ == '__main__':
    print('hyperparameter tuning using grid search')
    train_data = pd.read_csv('./dataset/train_preprocessed_final.csv')
    # balanced_data = pd.read_csv('./dataset/train_balanced.csv')
    y = train_data['Stay'].values
    x = train_data.drop('Stay',axis = 1)
    x = pd.get_dummies(x)
    param_test1 = {
        'max_depth':range(3,10,2),
        'min_child_weight':range(1,6,2)}
    result_df = grid_search(param_test1,x,y)
    result_df.to_csv('./dataset/hyper_parameter.csv',index = False)
