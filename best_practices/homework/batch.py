#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd


def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def main(year, month):

    

    # File input
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/yellow_tripdata_{year:04d}-{month:02d}.parquet'

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load model
    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    # create categorical values
    categorical = ['PULocationID', 'DOLocationID']

    # Read and process the data
    df = read_data(input_file, categorical)
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    print('predicted mean duration:', y_pred.mean())


    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred


    df_result.to_parquet(output_file, engine='pyarrow', index=False)

    return df_result




if __name__ == "__main__":

    if len(sys.argv) < 3:
        print("Usage: batch.py <year> <month>")
    try: 
        # command line input
        year = int(sys.argv[1])
        month = int(sys.argv[2])
    except ValueError:
        print("Year and month must be integers.")
        sys.exit(1)

    main(year, month)
