'''
------------------------------------------Importing the Libraries--------------------------------------------
'''
print('Importing the Libraries Started')
import numpy as np
import pandas as pd
import swifter
import os
from functions import *
from sklearn.model_selection import train_test_split
import math
from imblearn.over_sampling import SMOTE 
import pickle

'''
-----------------------------------------Setting the Home Directory--------------------------------------------
'''
print('Setting the Home Directory Started')
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

'''
------------------------------------------Importing the Dataseet--------------------------------------------
'''
print('Importing the Dataseet Started')
train_agg_data = pd.read_csv('train_data/AggregateData_Train.csv',na_values='?')
train_trn_data = pd.read_csv('train_data/TransactionData_Train.csv')
test_agg_data  = pd.read_csv('test_data/AggregateData_Test.csv')
test_trn_data  = pd.read_csv('test_data/TransactionData_Test.csv')


def fun_tran_amt(data):
    return (-1 if data['C5'] == 'D' else 1) * data['C12']



train_trn_data['tran_amt'] = train_trn_data.apply(fun_tran_amt,axis = 1)

test  = pd.read_csv('test.csv')
test_bkp = test
data  = pd.concat([train,test],axis = 0).reset_index()
pd.options.display.float_format = '{:.4f}'.format 
'''
----------------------------------------Checking for data details--------------------------------------------
'''
print('Checking for data details Started')
#print(data.isnull().sum())  #To Check missing values per column
#print(data.describe())      #To Check the Descriptive Statistics of the Data
#print(data.dtypes)          #To Check the Data type of each column


'''
----------------------------------------Missing values Treatment--------------------------------------------
'''
print('Missing values Treatment Started')