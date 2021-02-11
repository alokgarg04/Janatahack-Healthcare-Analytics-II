# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 18:28:56 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score
from sklearn.model_selection import train_test_split
import model_training as mt
from sklearn.metrics import roc_auc_score


def evaluate_model(X, y):
    clf = RandomForestClassifier(n_estimators = 200,verbose= 4,random_state=42)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    scores = cross_val_score(clf, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores
def batch_prediction_cv(model,data = None):
    #  predict 10k at one time
    print('batch prediction started')
    predicted_values = []
    batch = len(data) - len(data)%10000
    for i in range(0,batch,10000):
        # predicted_values.extend((model.predict_proba(test_data[i:i+1000]))[:,1])
        predicted_values.append(model.predict_proba(data[i:i+10000]))
    if len(data)%10000  != 0:
        # last_batch = len(test_data)%10000
        # predicted_values.extend((model.predict_proba(test_data[batch:]))[:,1])
        predicted_values.append(model.predict_proba(data[batch:]))

    return np.array(predicted_values)


def validate_model(x_train,x_cv,y_train,y_cv):
    train_score = []
    cv_score = []
    esimators = [50,100]
    for estimator in esimators:
        clf = RandomForestClassifier(n_jobs = -1,n_estimators = estimator,verbose= 4,random_state=42)
        clf.fit(x_train,y_train)

        predict_train = batch_prediction_cv(clf, data = x_train)
        predict_cv = batch_prediction_cv(clf,data = x_cv)
        # print(predict_cv)

        score_train = roc_auc_score(y_train, predict_train.astype(np.float64),average= macro)
        train_score.append(score_train)

        score_cv = roc_auc_score(y_cv, predict_cv.astype(np.float64),average = macro )
        cv_score.append(score_cv)
    return train_score,cv_score











if __name__ == '__main__':
    print('using cross validation')
    # training_data = pd.read_csv('./dataset/data_updated_new.csv')
    # balanced_data = pd.read_csv('./dataset/train_balanced.csv')
    encoded_data_new = pd.read_csv('./dataset/data_encoded_new.csv')
    y = encoded_data_new['Stay'].values
    x = encoded_data_new.drop('Stay',axis = 1)
    x = pd.get_dummies(x)
    # model = RandomForestClassifier(n_estimators=200, class_weight='balanced',verbose=4)
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.2,stratify=y)
    x_train,x_cv,y_train,y_cv = train_test_split(x_train, y_train, random_state=42, test_size=0.2,stratify=y_train)

    # scores_cv = evaluate_model(x, y, model)
    # print(np.mean(scores_cv))
    # train_score,cv_score = validate_model(x_train,x_cv,y_train,y_cv)
    score_cv_rf = evaluate_model(x, y)




