# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 14:15:02 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
import data_pre_processing as dp
from sklearn.preprocessing import OrdinalEncoder



# =============================================================================
# def fill_null_values(data):
#     data.fillna(data['Bed Grade'].mode()[0])
#     return data
# =============================================================================


# =============================================================================
# def average_cost_perAge(data):
#     data['Age'] = data['Age'].astype('str')
#     data['age of patients'] = data['Age'].apply(lambda x : int(x.split('-')[1]))
#     data['age of patients'] = data['age of patients'].apply(lambda x : int(x) - int(5))
#     data['avg spent on age'] = round(data['Admission_Deposit']/data['age of patients'],2)
#     data.drop(['Age','Admission_Deposit','age of patients'],axis = 1, inplace=True)
#     return data
# =============================================================================



if __name__ == '__main__':
    print('preprocessing the test data')
    test = pd.read_csv('./dataset/test.csv')
    test_columns = test.columns
    nullvalues_test = dp.check_nullValues(test)
    # cols_used = ['case_id','Hospital_code','Hospital_type_code','Hospital_region_code','Available Extra Rooms in Hospital','Department','Ward_Type','Bed Grade','patientid','Type of Admission','Severity of Illness','Visitors with Patient','Age','Admission_Deposit']
    cols_used = ['case_id','Hospital_code','Hospital_type_code','Hospital_region_code','Available Extra Rooms in Hospital','Department','Ward_Type','Ward_Facility_Code','Bed Grade','patientid','Type of Admission','Severity of Illness','Visitors with Patient','Age','Admission_Deposit']
    test_data_1 = test[cols_used]
    nullvalues_test_1 = dp.check_nullValues(test_data_1)
    test_data_2 = dp.fill_null_values(test_data_1)
    nullvalues_test_2 = dp.check_nullValues(test_data_1)
    test_updated = dp.average_cost_perAge(test_data_2)

    cols_toLabelENcode = ['Hospital_type_code','Hospital_region_code','Department','ward rating','critical']
    x1 = OrdinalEncoder().fit_transform(test_updated[cols_toLabelENcode])
    x1 = pd.DataFrame(data = x1, columns = cols_toLabelENcode)
    cols_NotLabelENcode = ['case_id','Hospital_code','Available Extra Rooms in Hospital','Bed Grade','Visitors with Patient','Admission_Deposit','age of patients','avg spent on age','number of Department visisted','mean deposite per paitient','avg_cost_critical','cost_per_department']
    x2 = test_updated[cols_NotLabelENcode]
    test_preprocessed = pd.concat([x1,x2],axis =1 )
    # test_preprocessed = pd.get_dummies(test_updated)
    test_preprocessed.to_csv('./dataset/test_preprocessed_ordinal.csv',index=False)
