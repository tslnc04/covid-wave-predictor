#!/usr/local/bin/python3.8

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import layers, callbacks, Input
from tensorflow import config

# Prevents some errors that would otherwise happen without this code
physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], enable=True)

import time
import os
import json
from io import StringIO

import numpy as np
import matplotlib.pyplot as plt

# plot_history uses the provided history and generates a plot of training and
# validation accuracy and loss vs epochs, saving it to the provided file paths
def plot_history(history, acc_name='keras_accuracy.png', loss_name='keras_loss.png'):
    # Make sure that the plot is clean before starting
    plt.clf()
    
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    plt.plot(epochs, accuracy, 'tab:blue', label = 'Training Accuracy')
    plt.plot(epochs, val_accuracy, 'tab:orange', label = 'Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(acc_name)

    # Avoid overwriting the plot just made
    plt.clf()

    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'tab:blue', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'tab:orange', label = 'Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(loss_name)

# File containing the preprocessed data as a dictionary with county hashes as
# keys and csv data as values
county_file = 'counties_data.json'

with open(county_file, 'r') as f:
    counties_data = json.load(f)

# Since the input data is biased with only ~31.38% having an output value of 1,
# the windows are put into two different lists depending on the value of the
# output. Then the zeroes are shuffled, added to the ones, everything is
# shuffled, and then the arrays are concatenated together into a 3d numpy array
windows_zero = []
windows_one = []

# Create a StringIO for each of the csv strings and use numpy to load into an array
for county_data in counties_data.values():
    data = np.loadtxt(StringIO(county_data), delimiter=',')
    # The last 14 days of data are ignored since the retroactively applied
    # labels for whether there is an imminent wave need 14 days. The last 7 will
    # eventually be used for prediction.

    county_windows_zero = []
    county_windows_one = []
    for i in range(len(data)-28):
        single_window = data[i:i+14]
        if single_window[-1, -1]:
            county_windows_one.append(single_window)
        else:
            county_windows_zero.append(single_window)
    
    if len(county_windows_one) > 0:
        windows_one.append(np.array(county_windows_one))
    if len(county_windows_zero) > 0:
        windows_zero.append(np.array(county_windows_zero))

windows_zero = np.concatenate(windows_zero)
windows_one = np.concatenate(windows_one)


# Shuffle and join such that the original order is not preserved
np.random.shuffle(windows_zero)
windows = np.concatenate((windows_one, windows_zero[:len(windows_one)]))
np.random.shuffle(windows)

# Strip the outputs from input data and take only the output data from the last
# day of each window. Other days are covered by other windows.
windows_x, windows_y = windows[:,:,:-1], windows[:,-1,-1]

# Use a GRU since the accuracy is nearly the same as with an LSTM but slightly
# more efficient for training
model = Sequential([
    Input(shape=windows_x.shape[1:]),
    layers.LSTM(256, return_sequences=True),
    layers.LSTM(128),
    layers.Dropout(0.5),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])

# Load from the checkpoint file if it exists, otherwise compile the model as
# defined. Future checkpoints will also be saved to this path.
checkpoint_filepath = 'neural_checkpoint.h5'

if os.path.exists(checkpoint_filepath):
    model = load_model(checkpoint_filepath)
else:
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Callback to save the model whenever validation accuracy improves
checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath, monitor='val_accuracy', verbose=0, save_best_only=True, save_weights_only=False, mode='min')

# Custom callback to keep a history as the model trains, generating a new plot
# every 10 epochs
class PlotterCallback(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
    
    def on_epoch_end(self, epoch, logs={}):
        for k, v in logs.items():
            self.history[k].append(v)
        
        if (epoch + 1) % 10 == 0:
            plot_history(self.history, acc_name='plots/neural_accuracy.png', loss_name='plots/neural_loss.png')

plotter = PlotterCallback()

callbacks_list = [checkpoint, plotter]

# Time the training out of curiosity
start = time.time()
# Fit the model using the input and output data and letting keras take care of
# validaiton vs training data. Larger batch sizes are used to reduce some of the
# workload of constantly updating weights
history = model.fit(windows_x, windows_y, epochs=25, batch_size=64, validation_split=0.2, callbacks=callbacks_list)
end = time.time()

# Generate a final plot since the last one made by the callback could be a few
# epochs out of date
plot_history(history.history, acc_name='plots/neural_accuracy.png', loss_name='plots/neural_loss.png')

# print('#' * 80)
# print(f'Elapsed Training Time: {end - start:.02f}s')
# print('#' * 80)
