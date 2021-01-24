#!/usr/local/bin/python3.8

import csv
import json
from io import StringIO

import urllib.request

# Link to the NYT data
data_url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv'

# JSON file to output county data to once preprocessed
data_output = 'counties_data.json'

# Definition of a wave of COVID cases. If the SMA(7) of daily new cases in 14
# days is greater than the SMA(7) at a given time multiplied by epsilon, that
# counts as an impending wave.
epsilon = 1.05

# Dictionary where raw data will be placed using the SHA256 hash of the county
# and state as a key.
county_data = {}

# Loop for reading the raw data
with urllib.request.urlopen(data_url) as csvurl:
    csvdata = csvurl.read().decode()
    csvreader = csv.reader(StringIO(csvdata))
    next(csvreader)  # skip the header row
    for row in csvreader:
        # Generate the ID used for creating files and as a key in `county_data`
        county_id = f'{row[1]}|{row[2]}|{row[3]}'
        # Extract the deaths and cases from a row and conver to ints
        # Because the deaths column may be an empty string, it must be replaced
        # with a 0 so the int conversion doesn't fail.
        row_data = [int(row[5]) if row[5] else 0, int(row[4])]
        # Add the data to `county_data`, creating the key if it does not exist
        if county_id in county_data:
            county_data[county_id].append(row_data)
        else:
            county_data[county_id] = [row_data]

# Dictionary to hold the processed county data as a csv string with the county
# hash as a key
processed_county_data = {}

# Testing to see if there is a bias to the data
will_increase_values = []

# Loop for processing the data together and writing it to disk
for county_id, data in county_data.items():
    # Data with less than 7 days won't work because of the moving average, but
    # data that's shorter in general isn't worth using since at least 21 days
    # are needed to actually provide training data.
    if len(data) < 30:
        continue

    # Calculate the change in deaths from the previous day. Since it will
    # occasionally be negative, clip values less than 0
    new_deaths = [data[0][0]] + [max(data[i+1][0] - data[i][0], 0) for i in range(len(data)-1)]
    # Apply the same clipping as with deaths, it might not be necessary
    new_cases = [data[0][1]] + [max(data[i+1][1] - data[i][1], 0) for i in range(len(data)-1)]
    # Calculate the 7-day simple moving average of daily new COVID cases
    cases_sma7 = [sum(new_cases[i:i+7])/7 for i in range(len(new_cases)-7)]
    # Uses `epsilon` to calculate whether there is a wave in the 14 days
    # following. See the comments on `epsilon` for more detail on the
    # calculation
    will_increase = [int(cases_sma7[i+14]/(cases_sma7[i] if cases_sma7[i] else 1) > epsilon) for i in range(len(cases_sma7)-14)]
    will_increase_values += will_increase
    # Fill in the last two weeks since this data won't be used for training,
    # only for the predictions.
    will_increase += [0] * 14
    
    # Take all of the data produced by the calculations above and create a
    # string representing CSV data to be written to the county file
    calculated_data = [f'{round(row[0], 4)},{round(row[1], 4)},{round(row[2], 4)}, {row[3]}' for row in zip(new_deaths[7:], new_cases[7:], cases_sma7, will_increase)]

    processed_county_data[county_id] = '\n'.join(calculated_data)

# Finally write the processed county data to a file
with open(data_output, 'w') as f:
    json.dump(processed_county_data, f)

# Display the percent that have ones to get a feel for the bias
# print(f'Percent of data with imminent COVID wave: {sum(will_increase_values) / len(will_increase_values)}')
# print('Do not worry too much about that percent, the training program automatically corrects biases')
