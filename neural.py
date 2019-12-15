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


# %% Create train dataset
NOISE_RATE = 30

with open('data/train-7-middle-item-as-label-with-noise-2.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for file in glob.glob('train/*'):
        mid = MidiFile(file)
        vector = []
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'note'):
                    vector.append(msg.note)

        noises = []
        for i in range(len(vector) - 6):
            window = vector[i: i + 7]
            train = window[:3] + window[4:]
            label = window[3]
            csv_writer.writerow(train + [label])

            # Add noisy data
            for j in range(3):
                seed(time.clock())
                train[j + 3] = 128 if train[j + 3] + randint(-NOISE_RATE, NOISE_RATE) > 128 else 0 if train[j + 3] + randint(-NOISE_RATE, NOISE_RATE) < 0 else train[j + 3] + randint(-NOISE_RATE, NOISE_RATE)
            noises.append(train + [label])

        for noise in noises:
            csv_writer.writerow(noise)

# %% Load data
data = np.loadtxt('data/query-7-middle-item-as-label.csv', delimiter=',')
X = data[:, :6]
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




# %% Evaluation
# _, accuracy = model.evaluate(X, label[:len(X)])
# print('Accuracy: %.2f' % (accuracy*100))


