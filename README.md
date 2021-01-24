# covid-wave-predictor

Predicts incoming waves in COVID-19 cases at the county level in the U.S.

## What?

Using the NYT COVID data, the risk of an imminent COVID wave in the next 14 days
is predicted. Also included is a basic webpage to allow the visualization of the
predictions for each county.

An imminent COVID wave is defined as a 5% increase in the 7-day moving average
of daily new cases 14 days after a given date.

## Usage

A demo of the website is available on
[Netlify](https://covid-wave-predictor.netlify.app). Additionally, a pre-trained keras
model is provided on GitHub at
[`TrainedModel.h5`](https://github.com/tslnc04/covid-wave-predictor/blob/master/TrainedModel.h5).
To use this model for predictions rather than the one generated during training,
change the `model_filepath` variable in
[`predict.py`](https://github.com/tslnc04/covid-wave-predictor/blob/master/predict.py),
otherwise the prediction will fail.

Since the NYT data is updated every day, generating new predictions requires
running the following in the repo.

```bash
python3.8 preprocess.py
python3.8 predict.py
```

## Dependencies

These are what were used when developing the code. Other versions may work, but
are untested and cannot be assumed to work. Other dependencies are used for the
website, but those are loaded by the page since NodeJS is not used.

- Tensorflow v2.3.1 using CUDA
- Python 3.8.3

## Credits

Data from The New York Times, based on reports from state and local health
agencies. Available on GitHub at
[`nytimes/covid-19-data`](https://github.com/nytimes/covid-19-data).

Map converted from Census Bureau boundary files available on
[census.gov](https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html).

## License

Copyright 2021 Timothy Laskoski. Licensed under MIT.
