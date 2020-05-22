# To-do: Unit testing
from ie_bike_model import model
import pytest
import pandas as pd
import numpy as np
import datetime as dt


@pytest.fixture
def input_dict():
   dict = {
   "date": dt.datetime(2011, 1, 1, 10, 0, 0),
   "weathersit": 1,
   "temperature_C": 15.58,
   "feeling_temperature_C": 19.695,
   "humidity": 76.0,
   "windspeed": 16.9,}
   return dict

@pytest.fixture
def input_df():
    df = pd.DataFrame(input_dict, index=[0])
    return df


#load_process_training_data pytest
# no need to parametrize as there is nothing passed to the function
def test_load_process_training_data():
    assert isinstance(model.process_training_data(), pd.DataFrame),"Failed for given DataFrame"


#train_and_persist pytest
#parametrize iss tacked to allow for all possible combinations
@pytest.mark.parametrize('rand_state',[(42)])
@pytest.mark.parametrize('comp_fact',[False,True,1,2,3])
def test_train_and_persist(rand_state,comp_fact):
    assert isinstance(model.train_and_persist(rand_state,comp_fact),"WIP"),"Failed for given Parameters"


#check_and_retrieve pytest
@pytest.mark.parametrize('file',[None])
@pytest.mark.parametrize('from_pack',[True,False])
@pytest.mark.parametrize('rand_state',[42])
@pytest.mark.parametrize('comp_fact',[(True,False,1,2,3)])
def test_check_and_retrieve(file,from_pack,rand_state,comp_fact):
    assert isinstance(model.check_and_retrieve(file,from_pack,rand_state,comp_fact),"WIP"),"Failed for given Parameters"


#get_season pytest
@pytest.mark.parametrize('date',["2011-01-01","01-01-2011","2011 January 1st","1st Jan 2011"])
def test_get_season(date):
    assert isinstance(model.get_season(date),int),"Failed for given Date"


#process_new_observation pytest
@pytest.mark.parametrize('df',[(input_df)])
def test_process_new_observation(df):
    assert isinstance(model.process_new_observation(df),pd.Dataframe),"Failed for given DataFrame"


#predict pytest
@pytest.mark.parametrize('dict', [(input_dict)])
def test_predict(dict):
    assert  isinstance(model.predict(dict), int),"Does not produce Integer"
    assert  isinstance(model.predict(dict), float),"Does not produce Float"
