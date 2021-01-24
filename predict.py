#!/usr/local/bin/python3.8

from tensorflow.keras.models import load_model
from tensorflow import config

# Prevents some errors that would otherwise happen without this code
physical_devices = config.list_physical_devices('GPU')
config.experimental.set_memory_growth(physical_devices[0], enable=True)

import json

import numpy as np
from io import StringIO

# File containing the preprocessed data as a dictionary with county hashes as
# keys and csv data as values
county_file = 'counties_data.json'

with open(county_file, 'r') as f:
    counties_data = json.load(f)

# List containing the last 14 days of data from each county. Unlike in
# training, this list is not concatenated and instead just becomes a 3d numpy
# array since it already is a list of 2d numpy arrays.
windows = []

for county_data in counties_data.values():
    data = np.loadtxt(StringIO(county_data), delimiter=',')
    # Rather than have multiple windows from one county for training, the last 7
    # days of data is used in each county for prediction
    windows.append(np.array(data[-14:]))

windows = np.array(windows)
# The column used in training as an output is zero-filled for the input that
# gets used for predictions, so it is left off of the windows input data
windows_x = windows[:,:,:-1]

# File to load the model from. The assumption is made that the file exists.
model_filepath = 'neural_checkpoint.h5'
model = load_model(model_filepath)

model.summary()

predictions = model.predict(windows_x, batch_size=64)

# List with a dictionary for each county
county_predictions = []

# Break apart the ID and use it to provide geographic data associated with the
# prediction
for county_id, prediction in zip(counties_data.keys(), predictions[:,0].astype(float)):
    county_id = county_id.split('|')

    # Since the NYT data lumps all of NYC together, there has to be an exception
    # when generating predictions that breaks the city back apart into the
    # boroughs for the map
    if county_id[0] == 'New York City':
        county_predictions.append({
            'county': 'Bronx',
            'state': county_id[1],
            'fips': '36005',
            'prediction': prediction,
        })
        county_predictions.append({
            'county': 'Kings',
            'state': county_id[1],
            'fips': '36047',
            'prediction': prediction,
        })
        county_predictions.append({
            'county': 'New York',
            'state': county_id[1],
            'fips': '36061',
            'prediction': prediction,
        })
        county_predictions.append({
            'county': 'Queens',
            'state': county_id[1],
            'fips': '36081',
            'prediction': prediction,
        })
        county_predictions.append({
            'county': 'Richmond',
            'state': county_id[1],
            'fips': '36085',
            'prediction': prediction,
        })
    else:
        county_predictions.append({
            'county': county_id[0],
            'state': county_id[1],
            'fips': county_id[2],
            'prediction': prediction,
        })

# Predictions are saved in JSON format to this file
predictions_file = 'www/county_predictions.json'

with open(predictions_file, 'w') as f:
    json.dump(county_predictions, f)
