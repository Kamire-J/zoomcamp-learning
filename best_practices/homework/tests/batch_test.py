from datetime import datetime 
import batch
import pandas as pd 

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    # create categorical values
    categorical = ['PULocationID', 'DOLocationID']

    actual_df = batch.prepare_data(df.copy(), categorical)

    # Confirm if the number of rows is correct
    assert len(actual_df) == 2
    
    # Check presence of new 'duration' column
    assert 'duration' in actual_df.columns

