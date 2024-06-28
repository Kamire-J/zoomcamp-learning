from pathlib import Path 
from datetime import datetime 
import batch
import pandas as pd 
import pytest
import os

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

def create_parquet(data):
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']

    df = pd.DataFrame(data, columns=columns)

    output_file = f'output/test_file.parquet'

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df.to_parquet(output_file, engine='pyarrow', index=False)


categorical = ['PULocationID', 'DOLocationID']

def prepare_data(df, categorical):
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def test_prepare_data():
    df = pd.read_parquet('output/test_file.parquet')
    actual_result = prepare_data(df categorical=categorical)
    expected_result = batch.read_data(df, categorical=categorical)

    pd.testing.assert_frame_equal(actual_result.reset_index(drop=True), expected_result.reset_index(drop=True))

