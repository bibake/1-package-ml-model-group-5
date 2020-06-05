# test
import joblib
import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

pd.options.mode.chained_assignment = None


def load_process_training_data():
    """
    Retrieves the pandas DataFrame object from the package and performs necessary
    feature engineering and transformation to prepare it for training.
    """

    # DATA PREP
    mod_dir, _ = os.path.split(__file__)

    DATA_PATH = os.path.join(mod_dir + "/data/bike_sharing")

    data = pd.read_csv(DATA_PATH + "/hour.csv", index_col="instant", parse_dates=True)

    # Change dteday to date time
    data["dteday"] = pd.to_datetime(data["dteday"])

    # Resolving skewness
    data["windspeed"] = np.log1p(data["windspeed"])
    data["cnt"] = np.sqrt(data["cnt"])

    # FEATURE ENGINEERING # #
    # Rented during office hours
    data["IsOfficeHour"] = np.where(
        (data["hr"] >= 9) & (data["hr"] < 17) & (data["weekday"] == 1), 1, 0
    )

    # Rented during daytime
    data["IsDaytime"] = np.where((data["hr"] >= 6) & (data["hr"] < 22), 1, 0)

    # Rented during morning rush hour
    data["IsRushHourMorning"] = np.where(
        (data["hr"] >= 6) & (data["hr"] < 10) & (data["weekday"] == 1), 1, 0
    )

    # Rented during evening rush hour
    data["IsRushHourEvening"] = np.where(
        (data["hr"] >= 15) & (data["hr"] < 19) & (data["weekday"] == 1), 1, 0
    )

    # Rented during most busy season
    data["IsHighSeason"] = np.where((data["season"] == 3), 1, 0)

    # Binning temp, hum in 5 equally sized bins
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    data["temp_binned"] = pd.cut(data["temp"], bins).astype("category")
    data["hum_binned"] = pd.cut(data["hum"], bins).astype("category")

    # Convert the data type to category
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "IsOfficeHour",
        "IsDaytime",
        "IsRushHourMorning",
        "IsRushHourEvening",
        "IsHighSeason",
    ]
    for col in int_hour:
        data[col] = data[col].astype("category")

    data = data.drop(columns=["dteday", "atemp", "casual", "registered"])

    return data


def train_and_persist(persist=None, random_state=42, compression_factor=False):
    """
    Train a RandomForestRegressor model and persist it as a pkl object in the user's
    home directory (default) or as per the path specified in `persist`.
    `random_state` enables the user to set their own seed for reproducibility purposes.
    `compression_factor` sets the compression level when persisting the pkl object.
    """

    # Interrupt and return if the intended path for persisting is ill-defined or the compression factor is invalid
    pkl_path = [os.path.join(os.path.expanduser("~"), "model.pkl"), persist][
        bool(persist)
    ]
    path = [pkl_path, os.path.split(pkl_path)[0]][pkl_path[-4:] == ".pkl"]
    if not os.path.exists(path):
        print("Error: The specified path for persisting does not exist")
        print("path: {}".format(path))
        return None
    if [
        type(compression_factor) is int and int(compression_factor) > 9,
        type(compression_factor) is tuple and compression_factor[1] > 9,
    ][isinstance(compression_factor, tuple)]:
        print("Invalid compression factor: {}".format(compression_factor))
        return None

    # Load and process training data
    train = pd.get_dummies(load_process_training_data())

    # Separate the independent and target variable on training data
    X_train = train.drop(columns=["cnt"], axis=1)
    y_train = train["cnt"]

    # Grid search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid={
            "max_depth": [10, 40],
            "min_samples_leaf": [1, 2],
            "min_samples_split": [2, 5],
            "n_estimators": [200, 400],
        },
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=4,
    )

    grid_result = gsc.fit(X_train, y_train)

    # Retrieve the best estimator from the grid search
    model = gsc.best_estimator_

    # Dump the model as a pkl object
    pkl_path = [os.path.join(pkl_path, "model.pkl"), pkl_path][pkl_path[-4:] == ".pkl"]
    joblib.dump(model, pkl_path, compress=compression_factor)

    return model


def check_and_retrieve(
    file=None,
    persist=None,
    from_package=False,
    random_state=42,
    compression_factor=False,
):
    """
    Check if a pretrained model is already stored as a pkl object in the specified path (`file`)
        If so, return `model`.
        If not:
            If `from_package` is True, retrieve the built-in pre-trained model.
            If not, check the argument `persist`:
                If not None, call `train_and_persist` to retrieve a newly trained model, persisted in `persist`.
                If None, check if there is a model.pkl object in the user's home directory:
                    If so, retrieve it.
                    Else, call `train_and_persist` to retrieve a newly trained model, persisted in the user's
                    home directory.
    """

    if file or from_package:
        path = [
            os.path.join(os.path.split(__file__)[0], "trained_model/model.pkl"),
            file,
        ][bool(file)]
        try:
            with open(path, "rb") as f:
                model = joblib.load(f)
        except:
            print(
                "Error: Could not load pkl object {}".format(
                    ["from the package", "from the given path"][bool(file)]
                )
            )
            print("path: {}".format(path))
            if file:
                print(
                    "{}".format(
                        [
                            "No pkl file included in the path",
                            "Check the path leading to the pkl file",
                        ][file[-4:] == ".pkl"]
                    )
                )
            return None
    elif persist:
        model = train_and_persist(
            persist=persist,
            random_state=random_state,
            compression_factor=compression_factor,
        )
    else:
        try:
            with open(os.path.join(os.path.expanduser("~"), "model.pkl"), "rb") as f:
                model = joblib.load(f)
        except:
            model = train_and_persist(
                persist=persist,
                random_state=random_state,
                compression_factor=compression_factor,
            )

    return model


def get_season(date_to_convert):
    """
    Return the season associated to the year embedded in the `date_to_convert` DateTime object
    """
    d_year = date_to_convert.year
    seasons = [
        (1, dt.date(d_year, 12, 21), dt.date(d_year, 12, 31)),
        (1, dt.date(d_year, 1, 1), dt.date(d_year, 3, 19)),
        (2, dt.date(d_year, 3, 20), dt.date(d_year, 6, 21)),
        (3, dt.date(d_year, 6, 22), dt.date(d_year, 9, 21)),
        (4, dt.date(d_year, 9, 22), dt.date(d_year, 12, 20)),
    ]

    for i in seasons:
        if date_to_convert.date() >= i[1] and date_to_convert.date() <= i[2]:
            return i[0]


def process_new_observation(df):
    """
    Process the input pandas DataFrame to comply with the format used to train the regressor
    """
    # Return immediately if the `dteday` key does not have a datetime object as value
    if not isinstance(df.dteday[0], dt.datetime):
        return None

    # Apply feature engineering
    df["mnth"] = df.dteday[0].month
    df["hr"] = df.dteday[0].hour
    df["season"] = get_season(df.dteday[0])
    df["yr"] = [0, 1][df.dteday[0].year % 2 == 0]
    df["weekday"] = df.dteday[0].weekday()
    df["workingday"] = 1 if df.weekday[0] < 5 else 0
    df["temp"] = df.temp / 41
    df["hum"] = df.hum / 100
    df["windspeed"] = df.windspeed / 67
    df["windspeed"] = np.log1p(df.windspeed)
    # If the optional argument `holiday` is not provided by the user, set to 0
    df["holiday"] = df.holiday if "holiday" in df else 0

    df["IsOfficeHour"] = (
        1 if (df.hr[0] >= 9) and (df.hr[0] < 17) and (df.weekday[0] == 1) else 0
    )
    df["IsDaytime"] = 1 if (df.hr[0] >= 6) and (df.hr[0] < 22) else 0
    df["IsRushHourMorning"] = (
        1 if (df.hr[0] >= 6) and (df.hr[0] < 10) and (df.weekday[0] == 1) else 0
    )
    df["IsRushHourEvening"] = (
        1 if (df.hr[0] >= 15) and (df.hr[0] < 19) and (df.weekday[0] == 1) else 0
    )
    df["IsHighSeason"] = 1 if df.season[0] == 3 else 0

    # Filter and reorder columns
    df = df[
        [
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
            "temp",
            "hum",
            "windspeed",
            "IsOfficeHour",
            "IsDaytime",
            "IsRushHourMorning",
            "IsRushHourEvening",
            "IsHighSeason",
        ]
    ]

    # Apply binning
    bins = [0, 0.19, 0.49, 0.69, 0.89, 1]
    df["temp_binned"] = pd.cut(df["temp"], bins).astype("category")
    df["hum_binned"] = pd.cut(df["hum"], bins).astype("category")

    # Merge with the training dataset to facilitate dummification
    train_df = load_process_training_data()
    train_df = train_df.drop(columns="cnt")
    df = train_df.append(df, ignore_index=True)

    # Cast variables appropriately
    int_hour = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "IsOfficeHour",
        "IsDaytime",
        "IsRushHourMorning",
        "IsRushHourEvening",
        "IsHighSeason",
    ]

    for col in int_hour:
        df[col] = df[col].astype("category")

    # Apply dummmification
    df = pd.get_dummies(df)
    df = df.iloc[-1:]

    # Return the processed DataFrame comprising the input observation
    return df


def predict(
    parameters,
    file=None,
    persist=None,
    from_package=False,
    random_state=42,
    compression_factor=False,
):
    """
    Pass the values stored in the `parameters` dictionary to the appropriate functions for processing
    and retrieve the hourly bike rental rate from the trained regressor.
    If specified, `file` indicates the path where the model (pkl object) is stored for retrieval.
    `from_package` enables the user to directly retrieve the built-in pre-trained regressor.
    If the regressor is meant to be trained and persisted locally, `persist` optionally indicates
    the path for persisting the resulting `model.pkl` file.
    `random_state` enables the user to set their own seed for reproducibility purposes.
    `compression_factor` sets the compression level when persisting the pkl object.
    """

    # Load or train model
    model = check_and_retrieve(
        file=file,
        persist=persist,
        from_package=from_package,
        random_state=random_state,
        compression_factor=compression_factor,
    )

    # Return immediately if any sanity check is not satisfied
    if not model:
        return None

    # Process `parameters` dictionary
    try:
        # Convert to pandas DataFrame
        df = pd.DataFrame(parameters, index=[0])

        cols = [
            "date",
            "weathersit",
            "temperature_C",
            "feeling_temperature_C",
            "humidity",
            "windspeed",
        ]

        # Ensure correct keys
        if set(df.columns) != set(cols) and set(df.columns) != set(cols + ["holiday"]):
            print(
                "Error: Please pass a dictionary object to the `parameters` argument with the following keys: \n\
            ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed'[, 'holiday']]"
            )
            return None

        if "holiday" in df.columns and not isinstance(df.holiday[0], np.int64):
            print("ERROR: Optional key `holiday` must be a (binary) integer object")
            return None
        if (
            "holiday" in df.columns
            and isinstance(df.holiday[0], np.int64)
            and str(df.holiday[0]) not in "10"
        ):
            print("ERROR: Optional key `holiday` can only hold values 1 or 0")
            return None

        # Rename columns conveniently
        df.rename(
            columns={
                "date": "dteday",
                "temperature_C": "temp",
                "feeling_temperature_C": "atemp",
                "humidity": "hum",
            },
            inplace=True,
        )

    except Exception:
        # Ensure correct type
        print(
            "Error: Please pass a dictionary object to the `parameters` argument with the following keys: \n\
        ['date', 'weathersit', 'temperature_C', 'feeling_temperature_C', 'humidity', 'windspeed'[, 'holiday']]"
        )
        return None

    # Process the DataFrame
    df = process_new_observation(df)

    # Return if sanity check is not satisfied
    if df is None:
        print("Error: Please assign a datetime object to the `date` input key")
        return None

    # Feed the processed observation to the regressor and retrieve prediction
    pred = model.predict(np.array(df).reshape(1, -1))

    return pred[0]
