import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
import keras


# import random
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt

np.set_printoptions(precision=3)

ntrial = 500
ndraw = 100

rowsDefault = []
rowsCheating = []
trueparms = []

for i in range(ntrial):
    a = float(np.random.rand(1))
    b = float(np.random.rand(1))
    trueparms.append([a, b])
    rowsDefault.append(np.random.normal(a, b, ndraw).tolist())
    z = np.random.normal(a, b, ndraw - 2).tolist()
    z.extend([a, b])
    rowsCheating.append(z)

dfxDefault = pd.DataFrame(rowsDefault)
dfxCheating = pd.DataFrame(rowsCheating)
dfy = pd.DataFrame(trueparms)

from sklearn.model_selection import train_test_split

default = train_test_split(dfxDefault, dfy, test_size = 0.2)
cheating = train_test_split(dfxCheating, dfy, test_size = 0.2)
d = {'xtrain': 0, 'xtest': 1, 'ytrain': 2, 'ytest': 3}


# x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
annDefault = Sequential()
annDefault.add(Dense(units=50, kernel_initializer='uniform', activation='relu', input_dim=ndraw))
annDefault.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
annDefault.add(Dense(units=2, kernel_initializer='uniform'))
annDefault.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
annDefault.fit(default[d['xtrain']], default[d['ytrain']], batch_size=10, epochs=250)  # Train
weightsDefault = annDefault.layers[0].get_weights()[0]  # View weights

annCheating = Sequential()
annCheating.add(Dense(units=50, kernel_initializer='uniform', activation='relu', input_dim=ndraw))
annCheating.add(Dense(units=50, kernel_initializer='uniform', activation='relu'))
annCheating.add(Dense(units=2, kernel_initializer='uniform'))
annCheating.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mse'])
annCheating.fit(cheating[d['xtrain']], cheating[d['ytrain']], batch_size=10, epochs=250)  # Train
weightsCheating = annCheating.layers[0].get_weights()[0]  # View weights

avgDefault = np.mean(weightsDefault, axis=1)
avgCheating = np.mean(weightsCheating, axis=1)

# Predicting the Test set results
yPredDefault = annDefault.predict(default[d['xtest']])
yPredCheating = annCheating.predict(cheating[d['xtest']])

# Compare answers
plt.scatter(yPredDefault[:,0], default[d['ytest']][0])
plt.scatter(yPredCheating[:,0], cheating[d['ytest']][0])


plt.scatter(yPredDefault[:,1], default[d['ytest']][1])
plt.scatter(yPredCheating[:,1], cheating[d['ytest']][1])