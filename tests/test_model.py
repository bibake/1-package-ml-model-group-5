# To-do: Unit testing
from ie_bike_model import model
import pytest
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.ensemble import RandomForestRegressor


@pytest.fixture
def input_dict():
    dict = {
        "date": dt.datetime(2011, 1, 1, 10, 0, 0),
        "weathersit": 1,
        "temperature_C": 15.58,
        "feeling_temperature_C": 19.695,
        "humidity": 76.0,
        "windspeed": 16.9,
    }
    return dict


@pytest.fixture
def input_df():
    df = pd.DataFrame(input_dict, index=[0])
    return df


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
@pytest.mark.parametrize("comp_fact", [True, 1, 10])
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
    "date", [2011-01-01, 01-01-2011,2014-02-03, 08-02-2014]
)
def test_get_season(date):
    assert isinstance(model.get_season(date), int), "Failed for given Date"


# process_new_observation pytest
@pytest.mark.parametrize("df", [(input_df)])
def test_process_new_observation(df):
    assert isinstance(
        model.process_new_observation(df), pd.Dataframe
    ), "Failed for given DataFrame"


# predict pytest
@pytest.mark.parametrize("dict", [(input_dict)])
def test_predict(dict):
    assert isinstance(model.predict(dict), (int, float)), "Does not produce Number"
