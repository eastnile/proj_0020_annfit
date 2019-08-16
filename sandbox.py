
import numpy as np
import tensorflow as tf
import random as rn
import pandas as pd
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)


rn.seed(12345)

tf.set_random_seed(1234)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,
                             inter_op_parallelism_threads=1)

from keras import backend as K

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed



sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)








# import random
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import matplotlib.pyplot as plt
np.set_printoptions(precision=3)


ntrial = 500
ndraw = 100

rows = []
trueparms = [] 
for i in range(ntrial):
    a = float(np.random.rand(1))
    b = float(np.random.rand(1))
    trueparms.append([a, b])
    rows.append(np.random.normal(a, b, ndraw))

dfx = pd.DataFrame(rows)
dfy = pd.DataFrame(trueparms)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dfx, dfy, test_size = 0.2)


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

# Train ANN
ann.save_weights('w0') # save initial weights
weights0a, biases0a = ann.layers[2].get_weights() # view initial weights for output layer

# Fitting the ANN to the Training set
ann.fit(x_train, y_train, batch_size = 10, epochs = 25) # Train once
weights1, biases1 = ann.layers[2].get_weights() # View weights

ann.load_weights('w0') # Restore initial weights
weights0b, biases0b = ann.layers[2].get_weights() # Confirm initial weights are in place
ann.fit(x_train, y_train, batch_size = 10, epochs = 25) # Train second time
weights2, biases2 = ann.layers[2].get_weights() # View new weights

print(weights1)
print('jomama')
print(weights2)

# Predicting the Test set results
#y_pred = ann.predict(x_test)

# Compare answers
#plt.scatter(y_pred[:,0],y_test[0])
#plt.scatter(y_pred[:,1],y_test[1])

#from ann_visualizer.visualize import ann_viz;
#Build your model here
#ann_viz(ann)


