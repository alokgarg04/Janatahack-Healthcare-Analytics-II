# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 12:49:40 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder




def batch_prediction(model,test_data):
    predicted_values = []
    batch = len(test_data) - len(test_data)%10000
    for i in range(0,batch,10000):
        predicted_values.extend(model.predict(test_data[i:i+10000]))
    if len(test_data)%10000  != 0:
        predicted_values.extend(model.predict(test_data[batch:]))

    return predicted_values

def Randomforest(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.33,stratify= y)
    rf = RandomForestClassifier(n_jobs=-1,verbose=4,n_estimators=200)
    rf.fit(x_train, y_train)
    prediction = batch_prediction(rf,x_test) #rf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,rf
def logistic_regression(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.33,stratify= y)
    rf = LogisticRegression(verbose=4,n_jobs=-1,max_iter=200)
    rf.fit(x_train, y_train)
    prediction = batch_prediction(rf,x_test) #rf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,rf

def supportVectorMachine(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.33,stratify= y)
    rf = SVC(decision_function_shape='ovo',verbose = 4)
    rf.fit(x_train, y_train)
    prediction = batch_prediction(rf,x_test)  #rf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,rf

def naiveBayes(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.33,stratify= y)
    rf = GaussianNB()
    # rf = SVC(decision_function_shape='ovo',verbose = 4)
    rf.fit(x_train, y_train)
    prediction = batch_prediction(rf,x_test) #rf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,rf

# binary:logistic
# multi:softprob
def xgBoost(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.2,stratify = y)
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,verbosity=2,n_jobs = -1)
    xgb_model.fit(x_train,y_train)
    prediction = batch_prediction(xgb_model,x_test) #xgb_model.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,xgb_model

def xgBoost_fulldata(x,y):
    # x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42,verbosity=2,n_jobs = -1,max_depth = 9)
    xgb_model.fit(x,y)
    prediction = batch_prediction(xgb_model,x) #xgb_model.predict(x_test)
    score = accuracy_score(y, prediction)
    return prediction, score,xgb_model

def xgboost_dmatrix(x,y):
    x_train,x_test,y_train,y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    params = {}
    params['objective'] = "multi:softprob"
    params['eval_metric'] = 'accuracy'
    # params['eta'] = 0.02
    params['max_depth'] = 6
    d_train = xgb.DMatrix(x_train, label = y_train)
    d_test = xgb.DMatrix(x_test, label = y_test)
    watchlist = [(d_train,'train'),(d_test,'valid')]
    gbdt = xgb.train(params,d_train,400,watchlist)
    return gbdt





if __name__ == '__main__':
    print('traing the model')
    # training_data = pd.read_csv('./dataset/data_updated_new.csv')
    # encoded_data = pd.read_csv('./dataset/train_final_encoded.csv')
    # encoded_data_new = pd.read_csv('./dataset/train_preprocessed_final.csv')
    # updated_cols = encoded_data_new.columns
    # encoded_data_new_head = encoded_data_new.head()
    train_new = pd.read_csv('./dataset/train_y_updated.csv')
    # train_new_cols = train_new.columns
    # balanced_data = pd.read_csv('./dataset/train_balanced.csv')
    y = train_new['Stay'].values
    # y = LabelEncoder().fit_transform(y).astype('float64')
    x = train_new.drop(['case_id','patientid','City_Code_Patient','Stay'],axis = 1)
    # 'Type of Admission','Severity of Illness',
    # cols_toLabelENcode = ['Hospital_type_code','Hospital_region_code','Department','ward rating','critical']
    # x1 = OrdinalEncoder().fit_transform(x[cols_toLabelENcode])
    # x1 = pd.DataFrame(data = x1, columns = cols_toLabelENcode)
    # cols_NotLabelENcode = ['Hospital_code','Available Extra Rooms in Hospital','Bed Grade','Visitors with Patient','Admission_Deposit','age of patients','avg spent on age','number of Department visisted','mean deposite per paitient','avg_cost_critical','cost_per_department']
    # x2 = x[cols_NotLabelENcode]
    # x_final = pd.concat([x1,x2],axis =1 )
    # x_final.to_csv('./dataset/x_final_ordinal.csv',index = False)
    # x = pd.get_dummies(x)
    # cols_train_preprocessed = x.columnss
    # data_final.head()
    # prediction_rf, score_rf,model_rf = Randomforest(x_final,y)
    # prediction_lr, score_lr,model_lr = logistic_regression(x_final,y)
    # prediction_svm, score_svm = supportVectorMachine(x,y)
    # prediction_nb, score_nb,model_nb = naiveBayes(x,y)
    # prediction_xbg, score_xgb,model_xgb = xgBoost(x,y)
    prediction_xbg_full, score_xgb_full,model_xgb_full = xgBoost_fulldata(x,y)
    # joblib.dump(model_xgb,'./models/model_xgblast.pkl')
    # gbdt = xgboost_dmatrix(x_final,y)
    joblib.dump(model_xgb_full,'./models/model_xgb_full_try.pkl')


