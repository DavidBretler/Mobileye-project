import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pydot
import graphviz
from keras.models import Sequential
from keras.layers import Dense
from keras import losses
from keras import optimizers
from keras import metrics
import os
from tensorflow.keras.optimizers import Adam
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'

# a sequential model - each layer has one input tensor and one output tensor
# the output tensor is used as the input for the next layer

# we start with an empty model:
model = Sequential()


#
# # our simple computational graph has one layer - a dense layer: y = W*X+b
# # (X is the input, y is the output, W are the weights, b is the 'bias')
# # there is one input tensor that has two numbers - shape (2,)
# layer = Dense(1, input_shape=(2,), activation=None, name='simple_layer')
#
# # .... input_shape=(1080, 796, 3) ...
#
# # add the layer to the model:
# model.add(layer)
#
# print(model.summary())
#
# from keras.utils import *
#
# keras.utils.plot_model(model, to_file='image.png')
#
# # a single random input - 1 is the number of inputs, 2 is the shape of a single input
# X = np.random.rand(1, 2)
# # X = np.random.rand(1, 1080, 796, 3)
# print(X)
#
#
# # Data
# # -------------------
#
# # the "secret" function:
# def linear_function(X):
#     # X is shape (None, 2)
#     # W[0]=1, W[1]=2, b=1
#     return 1 * X[:, [0]] + 2 * X[:, [1]] + 1
#
#
# # Training data
#
# # we need more data to train a network - let's try 1000:
X_train = np.random.rand(1000, 2)
# # the mapping to the output - x_0 + 2*x_1
# y_train = linear_function(X_train)
#
# # Validation data
#
# # we would like to test the network's training by giving it data it hasn't "seen" before:
X_val = np.random.rand(100, 2)
# # the mapvalping to the output  is the same as before- x_0 + 2*x_1
# y_val = linear_function(X_val)
#
# print("X_train.shape: ", X_train.shape)
# print("y_train.shape: ", y_train.shape)
# print("X_val.shape: ", X_val.shape)
# print("y_val.shape: ", y_val.shape)
#
# # Loss, optimizer and metrics
#
#
# mse = losses.MeanSquaredError()  # can use inputs to configure MeanSquaredError, Adam, etc.
#
# rmse = metrics.RootMeanSquaredError()
#
# model.compile(loss=mse, metrics=[rmse])
# # some default inputs can be called with strings, e.g:
# #adam=keras.optimizers
# model.compile(loss=mse, metrics = [rmse])
#
# print("inputs: ", model.inputs)
# print("outputs: ", model.outputs)
#
# # train the model:
#
# # batch size - how many inputs to run in one forward pass:
# bs = 10
# # epochs - how many times to go over the entire data
# e = 5

# model.fit(X_train, y_train, epochs=e, batch_size=bs, validation_data=(X_val, y_val))

# the model trained for 5 epochs. How did it fare?

# forward pass - calculate the output of the net on the inputs from the validation set:
# y_predict_1 = model.predict(X_val)

# compare the predicted values and the actual output values (y_predict vs. y_val):

# plt.plot(y_predict_1, y_val, '.', ms=1)
# plt.plot([0, 3.5], [0, 3.5], '--', linewidth=1)
# plt.xlabel('predicted')
# plt.ylabel('target')
# plt.show()

# Not bad! But can be improved...

# model.fit(X_train, y_train, epochs=100, batch_size=bs, validation_data=(X_val, y_val))
#
y_predict_2 = model.predict(X_val)
#
# plt.plot(y_predict_1, y_val, '.', ms=1, label='predict_1')
# plt.plot(y_predict_2, y_val, '.', ms=1, label='predict_2')
# plt.xlabel('predicted')
# plt.ylabel('target')
# plt.legend()
#
# plt.plot(y_predict_1, y_val, '.', ms=1, label='predict_1')
# plt.plot(y_predict_2, y_val, '.', ms=1, label='predict_2')
# plt.plot([0, 3.5], [0, 3.5], '--', linewidth=1)
# plt.xlabel('predicted')
# plt.ylabel('target')
# plt.legend()
# plt.show()

#
# # That was easy! Linear = Easy; Easy = Not interesting. Let's try a non-linear function:

# a non-linear function:
def non_linear_function(X):
    return np.exp(X[:, [0]] + 2 * X[:, [1]]) + 1


# Training data

# we need more data to train a network - let's try 1000:
X_train = np.random.rand(1000, 2)
# the mapping to the output - x_0 + 2*x_1
y_train = non_linear_function(X_train)

# Validation data

# we would like to test the network's training by giving it data it hasn't "seen" before:
X_val = np.random.rand(100, 2)
# the mapvalping to the output  is the same as before- x_0 + 2*x_1
y_val = non_linear_function(X_val)

# print(X_val.shape)
#
model_2 = Sequential()

layer = Dense(1, input_shape=(2,), activation=None, name='simple_layer')
model_2.add(layer)
adam = Adam()
rmse = metrics.RootMeanSquaredError()
mse = losses.MeanSquaredError()
model_2.compile(loss=mse,optimizer=adam, metrics=[rmse])

model_2.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_val, y_val))

y_predict = model_2.predict(X_val)

plt.plot(y_predict, y_val, '.', ms=1, label='predict')
plt.plot([1, 12], [1, 12], '--', label='desired')
plt.xlabel('predicted')
plt.ylabel('target')
plt.legend()
plt.show()


#
# # Horrible! What is wrong?
# model_4 = Sequential()
#
# input_layer = Dense(1, input_shape=(2,), activation='relu', name='input_layer')
# model_4.add(input_layer)
# for i in range(10):
#     layer = Dense(1, input_shape=(1,), activation='relu', name='simple_layer_' + str(i))
#     model_4.add(layer)
#
# #adam = optimizers.Adam()
# rmse = metrics.RootMeanSquaredError()
# mse = losses.MeanSquaredError()
# model_4.compile(loss=mse,  metrics=[rmse])
#
# model_4.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_val, y_val))

# # Even worse! How can we correct?
# model_5 = Sequential()
#
# input_layer = Dense(16, input_shape=(2,), activation='relu', name='input_layer')
# model_5.add(input_layer)
# for i in range(10):
#     layer = Dense(16, input_shape=(16,), activation='relu', name='simple_layer_' + str(i))
#     model_5.add(layer)
# output_layer = Dense(1, input_shape=(16,), activation=None, name='output_layer')
# model_5.add(output_layer)
#
# #adam = optimizers.Adam()
# rmse = metrics.RootMeanSquaredError()
# mse = losses.MeanSquaredError()
# model_5.compile(loss=mse,  metrics=[rmse])
#
# history_5 = model_5.fit(X_train, y_train, epochs=100, batch_size=10, validation_data=(X_val, y_val))
#
# print(model_5.summary())
# keras.utils.plot_model(model_5, to_file='image_5.png')
#
# y_predict_5 = model_5.predict(X_val)
#
# plt.plot(y_predict_2, y_val, '.', ms=1, label='predict_3')
# plt.plot(y_predict_5, y_val, '.', ms=1, label='predict_5')
# plt.plot([1, 15], [1, 15], '--', linewidth=0.5, label='desired')
# plt.xlabel('predicted')
# plt.ylabel('target')
# plt.legend()
#
# plt.plot(history_5.history['loss'], label='train loss')
# plt.plot(history_5.history['val_loss'], label='val loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
#
# # demonstration of overfit:
# model_6 = Sequential()
#
# input_layer = Dense(64, input_shape=(2,), activation='relu', name='input_layer')
# model_6.add(input_layer)
# for i in range(5):
#     layer = Dense(16, input_shape=(64,), activation='relu', name='simple_layer_' + str(i))
#     model_6.add(layer)
# output_layer = Dense(1, input_shape=(64,), activation=None, name='output_layer')
# model_6.add(output_layer)
#
# #adam = optimizers.Adam()
# rmse = metrics.RootMeanSquaredError()
# mse = losses.MeanSquaredError()
# model_6.compile(loss=mse, metrics=[rmse])
#
# print(model_6.summary())
#
# history_6 = model_6.fit(X_train, y_train, epochs=250, batch_size=10, validation_data=(X_val, y_val))
#
# y_predict_6 = model_6.predict(X_val)
# y_predict_6_train = model_6.predict(X_train[:10])
#
# plt.plot(y_predict_6, y_val, '.', ms=1, label='predict_6_val')
# plt.plot(y_predict_6_train, y_train[0:10], '.', ms=1, label='predict_6_train')
# plt.plot([1, 15], [1, 15], '--', linewidth=0.5, label='desired')
# plt.xlabel('predicted')
# plt.ylabel('target')
# plt.legend()
#
# # how overfit looks like in the history: Train loss goes to zero, val loss stays high
# plt.plot(history_6.history['loss'], label='train loss')
# plt.plot(history_6.history['val_loss'], label='val loss')
# plt.legend()
# plt.show()
