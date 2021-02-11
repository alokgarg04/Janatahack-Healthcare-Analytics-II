# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 08:54:26 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
import joblib
import os


def prediction_values(model,test_data):
    predicted = model.predict(test_data)
    return predicted










if __name__ == '__main__':
    print('predicting on testset')
    # test = pd.read_csv('./dataset/test.csv')
    test_preprocessed = pd.read_csv('./dataset/test_y_updated.csv')
    # cols_test_preprocessed = test_preprocessed.columns
    submission = pd.read_csv('./dataset/sample_submission.csv')
    submission['case_id'] = test_preprocessed['case_id'].values
    # submission = pd.DataFrame(columns=['case_id','Stay'])
    # submission_cols = submission.columns
    test_new = test_preprocessed.drop(['case_id','patientid','City_Code_Patient'],axis = 1)
    model = joblib.load('./models/model_xgb_full_try.pkl')
    # f_names = model.get_booster().feature_names
    # submission['case_id'] = test_preprocessed['case_id'].values
    # submission['Stay'] = prediction_values(model,test_preprocessed.loc[:,f_names])
    target_dict = {0:'0-10',1:'11-20',2:'21-30',3:'31-40',4:'41-50',5:'51-60',6:'61-70',7:'71-80',8:'81-90',9:'91-100',10:'More than 100 Days' }
    submission['Stay'] = prediction_values(model,test_new)
    submission.to_csv('./dataset/submission_notmapped.csv',index = False)
    submission['Stay'] = submission['Stay'].map(target_dict)
    submission.to_csv('./dataset/submission_mapped.csv',index = False)









