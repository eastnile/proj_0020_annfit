#import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)

ntrial = 500
ndraw = 100

rows = []
trueparms = [] 
for i in range(ntrial):
    a = float(np.random.rand(1))
    b = float(np.random.rand(1))
    trueparms.append([a,b])
    rows.append(np.random.normal(a,b,ndraw))

dfx = pd.DataFrame(rows)
dfy = pd.DataFrame(trueparms)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.2)


import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
ann = Sequential()

# Adding the input layer and the first hidden layer
ann.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu', input_dim = ndraw))

# Adding the second hidden layer
ann.add(Dense(units = 50, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
ann.add(Dense(units = 2, kernel_initializer = 'uniform'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['mse'])

# Fitting the ANN to the Training set
ann.fit(X_train, y_train, batch_size = 10, epochs = 1000)

# Predicting the Test set results
y_pred = ann.predict(X_test)

# Compare answers
plt.scatter(y_pred[:,0],y_test[0])
plt.scatter(y_pred[:,1],y_test[1])

