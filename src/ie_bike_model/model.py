import joblib
import glob
import os
import pandas as pd
import numpy as np
import datetime as dt
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
pd.options.mode.chained_assignment = None # ABB: Why is this needed?


def load_process_training_data():
    """
    Receives pandas dataframe and performs necessary feature engineering
    and transformation to prepare it for training.
    """

    # # DATA PREP # #
    mod_dir, _ = os.path.split(__file__)

    DATA_PATH = os.path.join(mod_dir + '/data/bike_sharing')

    data = pd.read_csv(DATA_PATH + "/hour.csv",
                       index_col="instant", parse_dates=True)

    # Change dteday to date time
    data['dteday'] = pd.to_datetime(data['dteday'])

    # resolving skewness
    data["windspeed"] = np.log1p(data["windspeed"])
    data["cnt"] = np.sqrt(data["cnt"])

    # # FEATURE ENGINEERING # #
    # Rented during office hours
    data['IsOfficeHour'] = np.where((data['hr'] >= 9) & (
        data['hr'] < 17) & (data['weekday'] == 1), 1, 0)

    # Rented during daytime
    data['IsDaytime'] = np.where((data['hr'] >= 6) & (data['hr'] < 22), 1, 0)

    # Rented during morning rush hour
    data['IsRushHourMorning'] = np.where((data['hr'] >= 6) & (
        data['hr'] < 10) & (data['weekday'] == 1), 1, 0)

    # Rented during evening rush hour
    data['IsRushHourEvening'] = np.where((data['hr'] >= 15) & (
        data['hr'] < 19) & (data['weekday'] == 1), 1, 0)

    # Rented during most busy season
    data['IsHighSeason'] = np.where((data['season'] == 3), 1, 0)

    # binning temp, atemp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    data['temp_binned'] = pd.cut(data['temp'], bins).astype('category')
    data['atemp_binned'] = pd.cut(data['atemp'], bins).astype('category')
    data['hum_binned'] = pd.cut(data['hum'], bins).astype('category')

    # Convert the data type to category
    int_hour = ["season", "yr", "mnth", "hr", "holiday",
                "weekday", "workingday", "weathersit",
                "IsOfficeHour", "IsDaytime", "IsRushHourMorning",
                "IsRushHourEvening", "IsHighSeason"]
    for col in int_hour:
        data[col] = data[col].astype("category")

    # ABB: Removed the dummify if-clause
    data = pd.get_dummies(data)

    return data


def train_and_persist(random_state=42, compression_factor=False):
    """
    Train a RandomForestRegressor model and persist it as a pkl object.
    `random_state` enables the user to set their own seed for reproducibility purposes.
    `compression_factor` sets the compression level when persisting the pkl object.
    """

    # load and process training data
    data = load_process_training_data()

    # # MODELING # #
    train = data.drop(columns=['dteday', 'casual', 'atemp', 'registered', 'temp', 'hum'])

    # separate the independent and target variable on testing data
    X_train = train.drop(columns=['cnt'], axis=1)
    y_train = train['cnt']

    # grid search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid={'max_depth': [10, 40],
                    'min_samples_leaf': [1, 2],
                    'min_samples_split': [2, 5],
                    'n_estimators': [200, 400]},
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=4
    )

    grid_result = gsc.fit(X_train, y_train)

    #model = RandomForestRegressor(max_depth=gsc.best_params_['max_depth'],
    #                              min_samples_leaf=gsc.best_params_['min_samples_leaf'],
    #                              min_samples_split=gsc.best_params_[
    #                                  'min_samples_split'],
    #                              n_estimators=gsc.best_params_['n_estimators'],
    #                              random_state=random_state)
    #model.fit(X_train, y_train)
    model = gsc.best_estimator_

    # ABB: Persisting as .pkl requires changing .gitignore (includes .pkl initially)
    # ABB: As per the rubric, the pkl object needs to be saved in the user home directory
    # ABB: Consider adding the destination path as argument with default the user home directory
    pkl_path = os.path.join(os.path.expanduser("~"),"model.pkl") 
    joblib.dump(model, pkl_path, compress=compression_factor)

    return model


def check_and_retrieve(file=None, from_package=False, random_state=42, compression_factor=False):
    """
    Check if a pretrained model is already stored as a pkl object in the specified path (`file`)
        If so, return `model`.
        If not:
            If `from_package` is True, retrieve the built-in pre-trained model.
            If not, check if there is a pkl object in the user's home directory. 
            Else, call `train_and_persist` to retrieve a newly trained model.
    """

    if file or from_package:
        path = [os.path.split(__file__)[0] + 'trained_model/model.pkl', file][bool(file)]
        try:
            with open(path, 'rb') as f:
                model = joblib.load(f)
        except:
            print('Error: Could not load pkl object {}'.format(['from the package','in the given path'][bool(file)]))
            if file:
                filename = re.split('[\\\/]',file)[-1]
                print('{}'.format(['No pkl file included in the path',
                                   'Check the path leading to the pkl file'][filename[-4:] == '.pkl']))
            return None
    else:
        try:
            with open(os.path.join(os.path.expanduser("~"),"model.pkl"),'rb') as f:
                model = joblib.load(f)
        except:
            model = train_and_persist(random_state=random_state, compression_factor=compression_factor)
        
    return model


def get_season(date_to_convert):
    d_year = date_to_convert.year
    seasons = [
        (1, dt.date(d_year, 12, 21), dt.date(d_year, 12, 31)),
        (1, dt.date(d_year, 1, 1), dt.date(d_year, 3, 19)),
        (2, dt.date(d_year, 3, 20), dt.date(d_year, 6, 21)),
        (3, dt.date(d_year, 6, 22), dt.date(d_year, 9, 21)),
        (4, dt.date(d_year, 9, 22), dt.date(d_year, 12, 20))
    ]

    for i in seasons:
        if date_to_convert >= i[1] and date_to_convert <= i[2]:
            return i[0]


def process_new_observation(df):
    try:
        df['mnth'] = df.dteday[0].month
        df['hr'] = df.dteday[0].hour
        df['season'] = get_season(df.dteday[0])
        df['yr'] = df.dteday[0].year - 2011
        df['weekday'] = df.dteday[0].isoweekday()
        df['workingday'] = (1 if df.weekday[0] < 6 else 0)
        df['temp'] = (df.temp - (-8)) / (39 - (-8))
        df['atemp'] = (df.atemp - (-16)) / (50 - (-16))
        df['hum'] = df.hum / 100
        df['windspeed'] = df.windspeed / 67
        # replace '0' with result from holiday calendar
        df['holiday'] = (df.holiday if 'holiday' in df else 0)
        # '154' is mean for entire dataste. Could be more targetted.
        df['registered'] = (df.registered if 'registered' in df else 154)
        # '36' is mean for entire dataste. Could be more targetted.
        df['casual'] = (df.casual if 'casual' in df else 36)
        df['IsOfficeHour'] = (1 if (df.hr[0] >= 9) and (
            df.hr[0] < 17) and (df.weekday[0] == 1) else 0)
        df['IsDaytime'] = (1 if (df.hr[0] >= 6) and (df.hr[0] < 22) else 0)
        df['IsRushHourMorning'] = (1 if (df.hr[0] >= 6) and (
            df.hr[0] < 10) and (df.weekday[0] == 1) else 0)
        df['IsRushHourEvening'] = (1 if (df.hr[0] >= 15) and (
            df.hr[0] < 19) and (df.weekday[0] == 1) else 0)
        df['IsHighSeason'] = (1 if df.season[0] == 3 else 0)
        df["windspeed"] = np.log1p(df.windspeed)

    except:
        print('Feature engineering error')

    try:
        df = df[['dteday', 'season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'hum',
                 'windspeed', 'casual', 'registered', 'IsOfficeHour', 'IsDaytime', 'IsRushHourMorning', 'IsRushHourEvening', 'IsHighSeason']]

    except:
        print('Column reordering error')

    try:
        # bin temp, atemp, hum
        bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
        df['temp_binned'] = pd.cut(df['temp'], bins).astype('category')
        df['atemp_binned'] = pd.cut(df['atemp'], bins).astype('category')
        df['hum_binned'] = pd.cut(df['hum'], bins).astype('category')

    except:
        print('Binning error')

    try:
        train_df = load_process_training_data(dummify=False)
        train_df = train_df.drop(columns='cnt')
        df = train_df.append(df, ignore_index=True)

    except:
        print("Data merging error")

    try:
        int_hour = ["season", "yr", "mnth", "hr", "holiday",
                    "weekday", "workingday", "weathersit",
                    "IsOfficeHour", "IsDaytime", "IsRushHourMorning",
                    "IsRushHourEvening", "IsHighSeason"]

        for col in int_hour:
            df[col] = df[col].astype("category")

    except:
        print('Data type updating error')

    try:
        df = pd.get_dummies(df)
        df = df.iloc[-1:]

    except:
        print('Dummifying error')

    return df


def predict(parameters, file=None, from_package=False, random_state=42, compression_factor=False):
    """
    1. Receives dictionary of input parameters
    2. Processes the input data
    3. Passes the data onto the trained model
    4. Returns the number of expected users
    """

    # load or train model
    model = check_and_retrieve(file=file, from_package=from_package, random_state=random_state, compression_factor=compression_factor)

    # # Process Parameters # #
    try:
        # convert to pandas df
        df = pd.DataFrame(parameters, index=[0])

        # ensure correct key-value pairs
        # IMPROVEMENT: Allow 'holiday', 'casual' and 'registered' to be optional keys.
        if list(df.columns) != ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']:
            print("ERROR: Please pass a dictionary to the 'parameters' argument with the following keys in the order presented here: \n\
            ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']")

        df.rename(columns={'date': 'dteday', 'temperature_C': 'temp',
                           'feeling_temperature_C': 'atemp', 'humidity': 'hum'}, inplace=True)

    except Exception:
        # ensure correct parameters syntax
        print("ERROR: Please pass a dictionary to the 'parameters' argument with the following keys in the order presented here: \n\
        ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']")

    try:
        df = process_new_observation(df)
        train = df.drop(columns=['dteday', 'casual', 'atemp', 'registered', 'temp', 'hum'])
    except:
        print('Preprocessing error')

    pred = model.predict(np.array(train).reshape(1, -1))

    return pred[0]
    print(pred[0])