#%%
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

# %% Create train dataset
with open('data/train-prepared-7.csv', mode='w') as csv_file:
    csv_writer = csv.writer(csv_file)
    for file in glob.glob('train/*'):
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

# %% Load data
data = np.loadtxt('data/train-prepared-7.csv', delimiter=',')
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
model.add(Dense(56, activation='sigmoid'))
model.add(Dense(112, activation='relu'))
model.add(Dense(128, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# %% Fit model
model.fit(X, label, epochs=100, batch_size=64)

# %%
model.save('models/NN_first.h5')

# %% Evaluation
_, accuracy = model.evaluate(data, X)
print('Accuracy: %.2f' % (accuracy*100))

# %% Predicting
predictions = model.predict(data)
#
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


