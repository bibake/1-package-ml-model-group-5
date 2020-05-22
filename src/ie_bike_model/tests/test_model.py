import model
import pytest
import joblib
import glob
import os
import pandas as pd
import numpy as np
import datetime as dt

#column_names = ["date","weathersit", "temperature_C", "feeling_temperature_C","humidity","windspeed"]
#error_df = pd.DataFrame(columns = column_names)

@pytest.mark.parametrize("dict,result",
                         [(
                             {
                             "date": dt.datetime(2011, 1, 1, 10, 0, 0),
                             "weathersit": 1,
                             "temperature_C": 15.58,
                             "feeling_temperature_C": 19.695,
                             "humidity": 76.0,
                             "windspeed": 16.9,
                         },36)
                             ])
def test_predict(df,result):
    assert  model.predict(dict)==result,"Failed for given Parameters"

    """
    try:
        assert  model.predict(dict)==result,dict
    except AssertionError as x:
        df=pd.DataFrame.from_dict(x)
        error_df=error_df.append(df)
    """

#error_df
