# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 22:14:36 2020

@author: Alok Garg
"""
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib
import model_training as mt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



def cross_validation_score(x,y):
    xgb_model = xgb.XGBClassifier(objective="multi:softprob", random_state=42,verbosity=2,n_jobs = -1)
    cross_val_score_xgb = cross_val_score(xgb_model,x,y)
    print(cross_val_score_xgb.mean())
    return cross_val_score_xgb.mean()

def cross_val_score_rf(x,y):
    rf = RandomForestClassifier(n_jobs=-1,n_estimators=200,verbose=4)
    cross_val_score_rf = cross_val_score(rf,x,y)
    print(cross_val_score_rf.mean())
    return cross_val_score_rf.mean()

def voting(feature,target):
    x_train,x_test,y_train,y_test = train_test_split(feature, target, random_state=42, test_size=0.33,stratify= target)
    clf1 = RandomForestClassifier(n_jobs=-1,n_estimators=200,verbose=4)
    clf2 = xgb.XGBClassifier(objective="multi:softprob", random_state=42,verbosity=2,n_jobs = -1)
    eclf1 = VotingClassifier(estimators=[('rf', clf1), ('xgb_model', clf2)], voting='hard')
    eclf1.fit(x_train, y_train)
    prediction = mt.batch_prediction(eclf1,x_test) #rf.predict(x_test)
    score = accuracy_score(y_test, prediction)
    return prediction, score,eclf1




if __name__ == '__main__':
    print('doing voting classifier')
    encoded_data_new = pd.read_csv('./dataset/train_preprocessed_final.csv')
    # updated_cols = encoded_data_new.columns
    # encoded_data_new_head = encoded_data_new.head()
    # balanced_data = pd.read_csv('./dataset/train_balanced.csv')
    y = encoded_data_new['Stay'].values
    x = encoded_data_new.drop('Stay',axis = 1)
    # 'Type of Admission','Severity of Illness',
    cols_toLabelENcode = ['Hospital_type_code','Hospital_region_code','Department','ward rating','critical']
    x1 = OrdinalEncoder().fit_transform(x[cols_toLabelENcode])
    x1 = pd.DataFrame(data = x1, columns = cols_toLabelENcode)
    cols_NotLabelENcode = ['Hospital_code','Available Extra Rooms in Hospital','Bed Grade','Visitors with Patient','Admission_Deposit','age of patients','avg spent on age','number of Department visisted','mean deposite per paitient']
    x2 = x[cols_NotLabelENcode]
    x_final = pd.concat([x1,x2],axis =1 )
    # x_final.to_csv('./dataset/x_final_ordinal.csv',index = False)
    # cross_val_score_rf = cross_val_score_rf(x_final,y)

    # cross_val_score_xgb = cross_validation_score(x_final,y)
    prediction_voted, score_voted,voted_model = voting(x_final,y)
    joblib.dump(voted_model,'./models/voted_model.pkl')
