# To-do: Unit testing
from ie_bike_model import model
import pytest
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor


def input_dict():
    dct = {
        "date": dt.datetime(2011, 1, 1, 10, 0, 0),
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
    }
    return dct

@pytest.fixture(name="input_dict")
def input_dict_fixture():
    return input_dict()
    

def wrong_dict():
    wdct = {
        "date": '2011-01-01',
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
    }
    return wdct

@pytest.fixture(name="wrong_dict")
def wrong_dict_fixture():
    return wrong_dict()


def holiday_dict():
    hdct = {
        "date": dt.datetime(2011, 1, 1, 10, 0, 0),
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
        "holiday": 1,
    }
    return hdct

@pytest.fixture(name="holiday_dict")
def holiday_dict_fixture():
    return holiday_dict()


def input_df():
    df = pd.DataFrame.from_dict({
        "date": dt.datetime(2011, 1, 1, 10, 0, 0),
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
    })
    return df

@pytest.fixture(name="input_df")
def input_df_fixture():
    return input_df()


def wrong_df():
    wdf = pd.DataFrame.from_dict({
        "date": '2011-01-01',
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
    })
    return wdf

@pytest.fixture(name="wrong_df")
def wrong_df_fixture():
    return wrong_df()


# load_process_training_data pytest
# no need to parametrize as there is nothing passed to the function
def test_load_process_training_data():
    assert isinstance(
        model.load_process_training_data(), pd.DataFrame
    ), "Returned object is not a DataFrame"


# train_and_persist pytest
# parametrize is stacked to allow for all possible combinations
@pytest.mark.parametrize("persist", ["/foo/bar/nowhere", None])
@pytest.mark.parametrize("rand_state", [42])
@pytest.mark.parametrize("comp_fact", [True, 10])
def test_train_and_persist(persist, rand_state, comp_fact):
    assert isinstance(
        model.train_and_persist(persist, rand_state, comp_fact),
        (RandomForestRegressor, type(None)),
    ), "Returned object is not a RandomForestRegressor or NoneType"


# check_and_retrieve pytest
@pytest.mark.parametrize("file", [None, "/foo/bar/nowhere", "/foo/bar/any.pkl"])
@pytest.mark.parametrize("persist", [None])
@pytest.mark.parametrize("from_pack", [True, False])
@pytest.mark.parametrize("rand_state", [42])
@pytest.mark.parametrize("comp_fact", [True])
def test_check_and_retrieve(file, persist, from_pack, rand_state, comp_fact):
    assert isinstance(
        model.check_and_retrieve(file, persist, from_pack, rand_state, comp_fact),
        (RandomForestRegressor, type(None)),
    ), "Returned object is not a RandomForestRegressor or NoneType"


# get_season pytest
@pytest.mark.parametrize(
    "date", [dt.datetime(2011, 1, 1, 0, 0, 0)]
)
def test_get_season(date):
    assert isinstance(model.get_season(date), int), "Returned object is not int or NoneType"


# process_new_observation pytest
@pytest.mark.parametrize("df", [input_df_fixture, wrong_df_fixture])
def test_process_new_observation(df):
    assert isinstance(
        model.process_new_observation(df), (pd.Dataframe, type(None))
    ), "Returned object is not DataFrame or NoneType"


# predict pytest
@pytest.mark.parametrize("dct", [input_dict_fixture, wrong_dict_fixture, holiday_dict_fixture])
def test_predict(dct):
    assert isinstance(model.predict(dct), (float, type(None))), "Returned object is not float or NoneType"
