# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:08:30 2018

@author: Asus
"""

from sklearn.preprocessing import MinMaxScaler

x_train = train_trn_data.groupby(['C2','C6'])[["tran_amt"]].sum()
y_train = train_agg_data['V3'] 

x_train = pd.pivot_table(train_trn_data, index = ['C2'], 
                         columns = ['C6'], values = 'tran_amt',dropna=True,aggfunc = np.sum).reset_index()

data = train_agg_data.merge(x_train, left_on='V2', right_on='C2', how='inner')

data_take = data.copy()
data_take_f = data_take[data_take['V3'].notnull()].reset_index()
data_take_f = data_take_f[data_take_f.V3 < 50000]
x_train_data = data_take_f.iloc[:,55:]
y_train_data = data_take_f.V3
x_train_data = x_train_data.drop('C2',axis = 1)
scaler = MinMaxScaler()
x_train_data = scaler.fit_transform(x_train_data)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()


x_train_data = x_train_data.fillna(0)
lm.fit(x_train_data.values, y_train_data)
lm.coef_

import matplotlib.pyplot as plt
plt.scatter(y_train_data,lm.predict(x_train_data))