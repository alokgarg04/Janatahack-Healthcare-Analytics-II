# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 20:00:21 2020

@author: Alok Garg
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import joblib
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# =============================================================================
# def neural_net(x_train,y_train):
#     model = Sequential()
#     model.add(Dense(14,activation='softmax'))
#     model.add(Dropout(0.5))
#
#     model.add(Dense(9, activation= 'softmax'))
#     model.add(Dropout(0.5))
#
#     model.add(Dense(11))
#
#     model.compile(optimizer='adam',loss='categorical_crossentropy')
#
#     model.fit(x_train, y_train,epochs = 2)
#
# =============================================================================
    # loss='categorical_crossentropy'

def baseline_model():
    # Create model here
    model = Sequential()
    model.add(Dense(10, input_dim = 14, activation = 'relu'))
    model.add(Dense(8, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(9, activation = 'relu'))
    model.add(Dense(11, activation = 'softmax')) # Softmax for multi-class classification
    # Compile model here
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

    # model.fit(x_train, y_train, batch_size=100, epochs= 10, verbose=4)
    return model






if __name__ == '__main__':
    print('trying with tensor flow')
    encoded_data_new = pd.read_csv('./dataset/train_preprocessed_final.csv')
    encoded_data_head = encoded_data_new.head()
    y = encoded_data_new['Stay'].values
    y = LabelEncoder().fit_transform(y).astype('float64')
    x = encoded_data_new.drop('Stay',axis =1)
    x_final = OrdinalEncoder().fit_transform(x) #pd.get_dummies(x)

    # x_final_head = x_final[:2]
    x_final_array = MinMaxScaler().fit_transform(x_final) #x_final.values
    # X_embedded = TSNE(n_components=2).fit_transform(x_final_array[:5000])
    # X_embedded.shape
    # plt.plot(X_embedded)
    x_train,x_test,y_train,y_test = train_test_split(x_final_array, y,  test_size=0.33)

    # early_stop = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
    estimator = KerasClassifier(build_fn = baseline_model)
    print('done here')
    m = estimator.fit(x_final_array, y,validation_split = 0.33, epochs = 100)
    # m = estimator.fit(x_train, y_train,validation_data=((x_test,y_test)),batch_size=128,epochs= 10)
    history_df = pd.DataFrame(m.history)
    print('done')
    history_df[['loss','val_loss']].plot()
    # joblib.dump(m,'./models/neural_network.pkl')
    # kfold = KFold(n_splits = 3, shuffle = True, random_state = 42)

    # results = cross_val_score(estimator, x_train, y_train, cv = kfold)
    # Result
    # print("Result: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    # baseline_model(x_train,x_test,y_train,y_test)

    # y = encoded_data_new['Stay'].values.astype('float64')
    # y[:2]
    # x = encoded_data_new.copy()
    # 'Type of Admission','Severity of Illness',
    # cols_toLabelENcode = ['Hospital_type_code','Hospital_region_code','Department','ward rating','critical','Stay']
    # x1 = OrdinalEncoder().fit_transform(x[cols_toLabelENcode])
    # print(x1[0])
    # x1 = StandardScaler().fit_transform(x1)
    # x1 = pd.DataFrame(data = x1, columns = cols_toLabelENcode)
    # cols_NotLabelENcode = ['Hospital_code','Available Extra Rooms in Hospital','Bed Grade','Visitors with Patient','Admission_Deposit','age of patients','avg spent on age','number of Department visisted','mean deposite per paitient']
    # x2 = x[cols_NotLabelENcode]
    # x_final = pd.concat([x1,x2],axis =1).astype('float64')
    # y = x_final['Stay'].values
    # x_f = x_final.drop('Stay',axis = 1)
    # x_final_head = x_final.head()
    # x_final_head.dtypes
    # x_final = MinMaxScaler().fit_transform(x_final)
    # x_final = np.hstack((x1,x2))
    # print(x_final[0])
    # print(x_final.shape)
    # x_final = x_final.to_numpy(dtype='float64')
    # print(x_final.dtypes)
    # x_train,x_test,y_train,y_test = train_test_split(x_f, y, random_state=42, test_size=0.2,stratify = y)
    # x_train = np.array(x_train)
    # x_train_minmax = MinMaxScaler().fit_transform(x_train)
    # x_test_minmax = MinMaxScaler().fit_transform(x_test)
    # print(x_train_minmax[0])
    # print(len(x_train))
    # print(x_train.shape)
    # print(set(y))
    # print(type(x_train))
    # print(x_train[0])
    # x_train =tf.convert_to_tensor(x_train)
    # y_train = tf.convert_to_tensor(y_train)
    # type(x_train)
    # x_train[0].shape
    # neural_net(x_train_minmax,y_train)
    # x_train.shape
