# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 08:23:59 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np

'''hypothesis:
    1.Department ----> 	Department overlooking the case
    2. Type of Admission ----> Admission Type registered by the Hospital
    3. Severity of Illness ----> Severity of the illness recorded at the time of admission
    4. Age ----> Age of the patient'''




def basic_details(data):
    print(data.shape)
    print()
    # print(data.dtypes)
    data_types = data.dtypes
    return data_types



def check_nullValues(data):
    nullValues = data.isnull().sum()
    return nullValues

def fill_null_values(data):
    data.fillna(data['Bed Grade'].mode()[0],inplace=True)
    return data

def Outliers_count(data):
    Q1 = np.percentile(data['Admission_Deposit'],q=25)
    # print(Q1)
    Q3 = np.percentile(data['Admission_Deposit'],q=75)
    # print(Q3)
    IQR = Q3-Q1
    # print(IQR)
    low = Q1- 1.5 * IQR
    high = Q3 + 1.5 * IQR
    # print(low,high)
    outliers = data['Admission_Deposit'].loc[(data['Admission_Deposit'] <= low) | (data['Admission_Deposit'] > high )]

    return outliers.index

def remove_outliers(data,idx):
    data = data.drop(idx)
    return data

def average_cost_perAge(data):
    data['Age'] = data['Age'].astype('str')
    data['age of patients'] = data['Age'].apply(lambda x : int(x.split('-')[1]))
    data['age of patients'] = data['age of patients'].apply(lambda x : int(x) - int(5))
    data['avg spent on age'] = round(data['Admission_Deposit']/data['age of patients'],2)
    data['mean deposite per paitient'] = data['patientid'].map(data.groupby('patientid')['Admission_Deposit'].mean().to_dict())
    data['number of Department visisted'] = data['patientid'].map(data.groupby('patientid')['Department'].count().to_dict())
    data['critical'] = data['Type of Admission'] + data['Severity of Illness']
    # Ward_Facility_Code
    # avg cost based on critical issues

    data['avg_cost_critical'] = data['critical'].map(data.groupby('critical')['Admission_Deposit'].mean().to_dict())
    data['cost_per_department'] = data['Department'].map(data.groupby('Department')['Admission_Deposit'].mean().to_dict())
    data['ward rating'] = data['Ward_Type'] + data['Ward_Facility_Code']
    # data.groupby(''ward rating'')
    data.drop(['Age','patientid','Type of Admission','Severity of Illness','Ward_Type','Ward_Facility_Code'],axis = 1, inplace=True)
    return data

if __name__ == '__main__':
    print('data prprocessing')
    train = pd.read_csv('./dataset/train.csv')
    
    column_info = pd.read_csv('./dataset/train_data_dict.csv')
    features = train.columns
    data_10 = train.head(10)
    data_types = basic_details(train)
    useful_feature = ['Hospital_code','Hospital_type_code','Hospital_region_code','Available Extra Rooms in Hospital','Department','Ward_Type','Ward_Facility_Code','Bed Grade','patientid','Type of Admission','Severity of Illness','Visitors with Patient','Age','Admission_Deposit','Stay']
    data = train[useful_feature]
    data.drop_duplicates(inplace=True)
    nullValues = check_nullValues(data)
    # only Bed grade has null values.. 113 value ---> dataset is big we can drop this
    data = fill_null_values(data)
    nullValues_1 = check_nullValues(data)
    # data_types = basic_details(data)
    outliers = Outliers_count(data)
    # data_1 = remove_outliers(data,outliers)
    # data_1.to_csv('./dataset/train_updated.csv',index = False)
    data_final = average_cost_perAge(data)
    # data_final.drop_duplicates(inplace=True)
    data_final.to_csv('./dataset/train_preprocessed_final.csv',index = False)

