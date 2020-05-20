import joblib
import glob
import os
import pandas as pd
import numpy as np
import datetime as dt


def train_model():
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

    # # DATA PREP # #
    mod_dir, _ = os.path.split(__file__)

    DATA_PATH = os.path.join(mod_dir + '/data/bike_sharing')

    data = pd.read_csv(DATA_PATH + "/hour.csv",
                       index_col="instant", parse_dates=True)

    # creating duplicate columns for feature engineering -- remove
    data['hr2'] = data['hr']
    data['season2'] = data['season']
    data['temp2'] = data['temp']
    data['hum2'] = data['hum']
    data['weekday2'] = data['weekday']

    # Change dteday to date time
    data['dteday'] = pd.to_datetime(data['dteday'])

    # Convert the data type to eithwe category or to float
    int_hour = ["season", "yr", "mnth", "hr", "holiday",
                "weekday", "workingday", "weathersit"]
    for col in int_hour:
        data[col] = data[col].astype("category")

    # resolving skewness
    data["windspeed"] = np.log1p(data.windspeed)
    data["cnt"] = np.sqrt(data.cnt)

    # # FEATURE ENGINEERING # #
    # Rented during office hours
    data['IsOfficeHour'] = np.where((data['hr2'] >= 9) & (
        data['hr2'] < 17) & (data['weekday2'] == 1), 1, 0)
    data['IsOfficeHour'] = data['IsOfficeHour'].astype('category')

    # Rented during daytime
    data['IsDaytime'] = np.where((data['hr2'] >= 6) & (data['hr2'] < 22), 1, 0)
    data['IsDaytime'] = data['IsDaytime'].astype('category')

    # Rented during morning rush hour
    data['IsRushHourMorning'] = np.where((data['hr2'] >= 6) & (
        data['hr2'] < 10) & (data['weekday2'] == 1), 1, 0)
    data['IsRushHourMorning'] = data['IsRushHourMorning'].astype('category')

    # Rented during evening rush hour
    data['IsRushHourEvening'] = np.where((data['hr2'] >= 15) & (
        data['hr2'] < 19) & (data['weekday2'] == 1), 1, 0)
    data['IsRushHourEvening'] = data['IsRushHourEvening'].astype('category')

    # Rented during most busy season
    data['IsHighSeason'] = np.where((data['season2'] == 3), 1, 0)
    data['IsHighSeason'] = data['IsHighSeason'].astype('category')

    # binning temp, atemp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    data['temp_binned'] = pd.cut(data['temp2'], bins).astype('category')
    data['hum_binned'] = pd.cut(data['hum2'], bins).astype('category')

    # dropping duplicated rows used for feature engineering
    data = data.drop(columns=['hr2', 'season2', 'temp2', 'hum2', 'weekday2'])

    # dummify
    data = pd.get_dummies(data)

    # # MODELING # #
    train = data.drop(columns=['dteday', 'casual', 'atemp', 'registered'])

    # seperate the independent and target variable on testing data
    X_train = train.drop(columns=['cnt'], axis=1)
    y_train = train['cnt']

    # grid search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={'max_depth': [10, 40],
                    'min_samples_leaf': [1, 2],
                    'min_samples_split': [2, 5],
                    'n_estimators': [200, 400]},
        cv=5,
        scoring="r2",
        verbose=1,
        n_jobs=4
    )

    grid_result = gsc.fit(X_train, y_train)

    model = RandomForestRegressor(max_depth=gsc.best_params_['max_depth'],
                                  min_samples_leaf=gsc.best_params_['min_samples_leaf'],
                                  min_samples_split=gsc.best_params_[
                                      'min_samples_split'],
                                  n_estimators=gsc.best_params_['n_estimators'],
                                  random_state=random_state)
    model.fit(X_train, y_train)

    joblib.dump(model, "model.pkl")

    return model


def train_and_persist(model_path=None, filename=None, retrain_model=False, random_state=42):
    """
    Check if pretrained model exists.
    If so, return model.
    If not, train and save new RandomForestRegressor model.
    """

    if model_path:
        try:
            if filename:
                try:
                    model = joblib.load(model_path + filename)
                except:
                    print('fail_1')
            else:
                try:
                    model = joblib.load(model_path + 'model.pkl')
                except:
                    pass
                try:
                    model = joblib.load(model_path)
                except:
                    pass

        except Exception:
            print('fail_2')

    elif filename:
        try:
            model = joblib.load(filename)
        except Exception:
            print('fail_3')

    else:
        if glob.glob('model.pkl'):
            try:
                model = joblib.load('model.pkl')
            except Exception:
                print('fail_4')

        elif retrain_model:  # Train and save new model
            try:
                model = train_model()

            except Exception:
                print('fail_5')

        else:  # For pre-trained model included in package
            try:
                mod_dir, _ = os.path.split(__file__)

                MODEL_PATH = os.path.join(mod_dir + '/trained_model/model.pkl')

                model = joblib.load(MODEL_PATH)

                joblib.dump(model, "model.pkl")
            except Exception:
                print('fail_6')

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


def predict(parameters, model_path=None, filename=None, random_state=42):
    """
    1. Receives dictionary of input parameters
    2. Processes the input data
    3. Passes the data onto the trained model
    4. Returns the number of expected users
    """

    # load or train model
    model = train_and_persist(model_path=model_path, filename=filename, random_state=random_state)

    # # Process Parameters # #

    try:
        # convert to pandas df
        df = pd.DataFrame(parameters, index=[0])

        # ensure correct key-value pairs
        # IMPROVEMENT: Allow 'holiday', 'casual' and 'registered' to be optional keys.
        if list(df.columns) != ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']:
            print("ERROR: Please pass a dictionary to the 'parameters' argument with the following keys in the order presented here: \n\
            ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']")

        # rename columns –– may be able to remove this step
        df.rename(columns={'date': 'dteday', 'temperature_C': 'temp',
                           'feeling_temperature_C': 'atemp', 'humidity': 'hum'}, inplace=True)

    except Exception:
        # ensure correct parameters syntax
        print("ERROR: Please pass a dictionary to the 'parameters' argument with the following keys in the order presented here: \n\
        ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed']")

    try:
        df['mnth'] = df.dteday[0].month
        df['hr'] = df.dteday[0].hour
        df['season'] = get_season(df.dteday[0])
        df['yr'] = df.dteday[0].year - 2011
        df['weekday'] = df.dteday[0].weekday()
        df['workingday'] = (1 if df.dteday[0].weekday() < 6 else 0)
        df['temp'] = (df.temp - (-8)) / (39 - (-8))
        df['atemp'] = (df.atemp - (-16)) / (50 - (-16))
        df['hum'] = df.hum / 100
        df['windspeed'] = df.windspeed / 67
        # replace '0' with result from holiday calendar
        df['holiday'] = (df.holiday if 'holiday' in df else 0)
        # replace '0' with mean
        df['registered'] = (df.registered if 'registered' in df else 0)
        # replace '0' with mean
        df['casual'] = (df.casual if 'casual' in df else 0)

        # train = hour_train.drop(columns = ['dteday', 'casual','atemp', 'registered'])
    except:
        pass
    # processed_params

    # return prediction
    # return model.predict(np.array(processed_params).reshape(1, -1))
    print('done')
