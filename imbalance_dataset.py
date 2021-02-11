# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:17:17 2020

@author: Alok Garg
"""
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


def oversampling(x,y):
    ovs = SMOTE()
    x,y = ovs.fit_resample(x, y)
    return x,y















if __name__ == '__main__':
    print('balancing the dataset')
    dataset = pd.read_csv('./dataset/train.csv')
    training_data_imbalace = pd.read_csv('./dataset/train_preprocessed_final.csv')

    y = training_data_imbalace['Stay'].values
    x = training_data_imbalace.drop('Stay',axis = 1)
    x = pd.get_dummies(x)

# =============================================================================
#     y = training_data_imbalace['Stay'].values
#     x = training_data_imbalace.drop('Stay',axis = 1)
# =============================================================================
    x_updated,y_updated = oversampling(x,y)
    x_updated['Stay'] = y_updated
    x_updated.to_csv('./dataset/train_balanced_final.csv', index=False)
    # x_updated['Stay'] = y_updated
