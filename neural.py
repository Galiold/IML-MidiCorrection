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


# %% Create train dataset
NOISE_RATE = 30

with open('data/query-7-middle-item-as-label.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for file in glob.glob('validation/query/*'):
        mid = MidiFile(file)
        vector = []
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'note'):
                    vector.append(msg.note)

        for i in range(len(vector) - 6):
            window = vector[i: i + 7]
            train = window[:3] + window[4:]
            label = window[3]
            csv_writer.writerow(train + [label])

            #   Add noisy data
            # for j in range(3):
            #     seed(time.clock())
            #     train[j + 3] = 128 if train[j + 3] + randint(-NOISE_RATE, NOISE_RATE) > 128 else 0 if train[j + 3] + randint(-NOISE_RATE, NOISE_RATE) < 0 else train[j + 3] + randint(-NOISE_RATE, NOISE_RATE)
            # csv_writer.writerow(train + [label])

# %% Load data
data = np.loadtxt('data/query-7-middle-item-as-label.csv', delimiter=',')
X = data[:, :6]
Y = data[:, 6]

label = np.zeros((len(Y), 128))
for i, val in enumerate(Y):
    label[i, int(val)] = 1

label = label.astype('int16')


# %% Load Validation Data
data = np.loadtxt('data/groundTruth-7-middle-item-as-label.csv', delimiter=',')
# X = data[:, :6]
Y = data[:, 6]

label = np.zeros((len(Y), 128))
for i, val in enumerate(Y):
    label[i, int(val)] = 1

label = label.astype('int16')

# %% Create Model
model = Sequential()
model.add(Dense(7, input_dim=6, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(112, activation='relu'))
model.add(Dense(128, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% Fit model
model.fit(X, label, epochs=100, batch_size=64)

# %% Save model
model.save('models/NN_train-7-middle-item-as-label-with-noise.h5')

# %% Load model
model.load_weights('models/NN_first.h5')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# %% Evaluation
_, accuracy = model.evaluate(X, label)
print('Accuracy: %.2f' % (accuracy*100))

# %% Predicting
predictions = model.predict(X)
index = np.where(predictions[0] == np.max(predictions[0]))[0]
print(index[0])
#%%
rounded = [round(x[0]) for x in predictions]

for i, predict in enumerate(rounded):
    if predict != X[i]:
        print('predict: %d, value: %d' % (predict, X[i]))
# print(rounded)
#
# predictions = model.predict_classes(data)
#
# for i in range(5):
#     print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], X[i]))


