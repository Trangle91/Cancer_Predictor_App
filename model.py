# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 17:15:34 2019

@author: Omkar
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

dataset = pd.read_csv('data.csv')


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(max_depth=19, min_samples_split=50)
DT.fit(X, y)

pickle.dump(DT, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
print(model.predict([[55,3.0,5.2,2.9,0.0,1.0,0.0,1.0,0.0,0.0]]))

