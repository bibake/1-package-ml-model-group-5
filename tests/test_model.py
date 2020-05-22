# To-do: Unit testing
from ie_bike_model import model
import pytest
import pandas as pd
import numpy as np
import datetime as dt

#load_process_training_data pytest
@pytest.mark.parametrize('df',[()])
def test_load_process_training_data(df):
    assert isinstance(model.process_training_data(df), pd.DataFrame),"Failed for given DataFrame"


#train_and_persist pytest
@pytest.mark.parametrize('rand_state,comp_fact',[(42,True)])
def test_train_and_persist(rand_state,comp_fact):
    assert isinstance(model.train_and_persist(rand_state,comp_fact),"WIP"),"Failed for given Parameters"


#check_and_retrieve pytest
@pytest.mark.parametrize('file,from_pack,rand_state,comp_fact',[(None,False,42,True)])
def test_check_and_retrieve(file,from_pack,rand_state,comp_fact):
    assert isinstance(model.check_and_retrieve(file,from_pack,rand_state,comp_fact),"WIP"),"Failed for given Parameters"


#get_season pytest
@pytest.mark.parametrize('date',[("2011-01-01")])
def test_get_season(date):
    assert isinstance(model.get_season(date),"WIP"),"Failed for given Date"


#process_new_observation pytest
@pytest.mark.parametrize('df',[()])
def test_process_new_observation(df):
    assert isinstance(model.process_new_observation(df),pd.Dataframe),"Failed for given DataFrame"


#predict pytest
@pytest.mark.parametrize('dict',
                         [(
                             {
                             "date": dt.datetime(2011, 1, 1, 10, 0, 0),
                             "weathersit": 1,
                             "temperature_C": 15.58,
                             "feeling_temperature_C": 19.695,
                             "humidity": 76.0,
                             "windspeed": 16.9,
                         })
                             ])
def test_predict(df,result):
    assert  isinstance(model.predict(dict), int),"Does not produce Integer"
    assert  isinstance(model.predict(dict), float),"Does not produce Float"
