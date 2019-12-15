# %% Imports
import keras
import numpy as np
import mido
from keras import Sequential
from keras.layers import Dense
from mido import MidiFile
# import pandas as pd
# import os
import csv
import glob as glob
from random import seed, randint
import time
import matplotlib.pyplot as plt

# %% Load data from query
data = np.loadtxt('data/query-7-middle-item-as-label.csv', delimiter=',')
queryData = data[:, :6]
queryLbl = data[:, 6]

# %% Load label from ground truth
data = np.loadtxt('data/groundTruth-7-middle-item-as-label.csv', delimiter=',')
truthData = data[:, :6]
truthLbl = data[:, 6]

# label = np.zeros((len(Y), 128))
# for i, val in enumerate(Y):
#     label[i, int(val)] = 1
#
# label = label.astype('int16')

# %% Load model
model = Sequential()
model.add(Dense(7, input_dim=6, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(112, activation='relu'))
model.add(Dense(128, activation='softmax'))

model.load_weights('models/NN_train-7-middle-item-as-label.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% Predicting
predictions = model.predict(queryData)
# sum_of_exps = np.sum

replacements = []
for i, prediction in enumerate(predictions):
    index = np.where(predictions[i] == np.max(predictions[i]))[0]
    # print(index[0])
    replacements.append(index[0])

# for i, prediction in enumerate(replacements):
#     print(prediction, ' ', np.where(label[i] == 1)[0][0])

# %%
queries = []
for i in range(len(queryData)):
    queries.append(np.concatenate((queryData[i][:3], queryLbl[i], queryData[i][3:]), axis=None))

groundTruth = []
for i in range(len(queryData)):
    groundTruth.append(np.concatenate((truthData[i][:3], truthLbl[i], truthData[i][3:]), axis=None))

predicts = []
for i in range(len(queryData)):
    predicts.append(np.concatenate((queryData[i][:3], replacements[i], queryData[i][3:]), axis=None))

# %%
for i in range(200):
    plt.xlabel('Window')
    plt.ylabel('Notes')
    plt.plot(queries[i], label='Query', color='red')
    plt.plot(groundTruth[i], label='Ground Truth', linestyle='dashed', color='green')
    plt.plot(predicts[i], label='Prediction', linestyle='dotted', color='blue')

    plt.legend()
    # plt.show()
    plt.savefig('result/validation-normal/result%s.png' % i)
    plt.close()
    # print(queries)

