# To-do: Unit testing
from ie_bike_model import model
import pytest
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor

# Sample dictionaries
idct = {
    "date": dt.datetime(2011, 1, 1, 10, 0, 0),
    "weathersit": 1,
    "temperature_C": 15.58,
    "feeling_temperature_C": 19.695,
    "humidity": 76.0,
    "windspeed": 16.9,
}

wdct = {
    "date": "2011-01-01",
    "weathersit": 1,
    "temperature_C": 15.58,
    "feeling_temperature_C": 19.695,
    "humidity": 76.0,
    "windspeed": 16.9,
}

hdct = {
    "date": dt.datetime(2011, 1, 1, 10, 0, 0),
    "weathersit": 1,
    "temperature_C": 15.58,
    "feeling_temperature_C": 19.695,
    "humidity": 76.0,
    "windspeed": 16.9,
    "holiday": 1,
}

# Sample dataframes
idf = pd.DataFrame(
    [
        {
            "dteday": dt.datetime(2011, 1, 1, 10, 0, 0),
            "weathersit": 1,
            "temp": 15.58,
            "atemp": 19.695,
            "hum": 76.0,
            "windspeed": 16.9,
        }
    ]
)

wdf = pd.DataFrame(
    [
        {
            "dteday": "2011-01-01",
            "weathersit": 1,
            "temp": 15.58,
            "atemp": 19.695,
            "hum": 76.0,
            "windspeed": 16.9,
        }
    ]
)


# load_process_training_data pytest
# no need to parametrize as there is nothing passed to the function
def test_load_process_training_data():
    assert isinstance(
        model.load_process_training_data(), pd.core.frame.DataFrame
    ), "Returned object is not pd.core.frame.DataFrame"


# train_and_persist pytest
@pytest.mark.parametrize(
    "persist, rand_state, comp_fact, result_instance",
    [
        ("/foo/bar/nowhere", 42, True, type(None)),
        (None, 42, True, RandomForestRegressor),
        (None, 42, 10, type(None)),
    ],
)
def test_train_and_persist(persist, rand_state, comp_fact, result_instance):
    assert isinstance(
        model.train_and_persist(persist, rand_state, comp_fact), result_instance,
    ), "Returned object is not {}".format(result_instance)


# check_and_retrieve pytest
@pytest.mark.parametrize(
    "file, persist, from_pack, rand_state, comp_fact, result_instance",
    [
        (None, None, True, 42, True, RandomForestRegressor),
        ("/foo/bar/nowhere", None, True, 42, True, type(None)),
        ("/foo/bar/any.pkl", None, True, 42, True, type(None)),
        (None, None, False, 42, True, RandomForestRegressor),
        ("/foo/bar/nowhere", None, False, 42, True, type(None)),
        ("/foo/bar/any.pkl", None, False, 42, True, type(None)),
    ],
)
def test_check_and_retrieve(
    file, persist, from_pack, rand_state, comp_fact, result_instance
):
    assert isinstance(
        model.check_and_retrieve(file, persist, from_pack, rand_state, comp_fact),
        result_instance,
    ), "Returned object is not {}".format(result_instance)


# get_season pytest
@pytest.mark.parametrize("date", [dt.datetime(2011, 1, 1, 0, 0, 0)])
def test_get_season(date):
    assert isinstance(model.get_season(date), int), "Returned object is not int"


# process_new_observation pytest
@pytest.mark.parametrize(
    "df, result_instance", [(idf, pd.core.frame.DataFrame), (wdf, type(None))]
)
def test_process_new_observation(df, result_instance):
    assert isinstance(
        model.process_new_observation(df), result_instance
    ), "Returned object is not {}".format(result_instance)


# predict pytest
@pytest.mark.parametrize(
    "dct, result_instance", [(idct, float), (wdct, type(None)), (hdct, float)]
)
def test_predict(dct, result_instance):
    assert isinstance(
        model.predict(dct), result_instance
    ), "Returned object is not {}".format(result_instance)
