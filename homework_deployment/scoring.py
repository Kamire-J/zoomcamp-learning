import pandas as pd 
import numpy as np 
import pickle

# Loading the model
def load_model_and_vectorizer():
    # Load DictVectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    return model, dv

# List of categorical columns
categorical = ['PULocationID', 'DOLocationID']

def read_data(year, month):
    # Construct the URL based on the year and month
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    
    # Read the Parquet file from the URL
    df = pd.read_parquet(url)
    
    # Calculate the duration
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    # Filter out unrealistic durations
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    
    # Handle missing values and convert to string type
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    # Create ride_id column
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    
    return df 

# Transform, predict
def score_data(df, model, dv):
    # Convert the DataFrame to a dictionary of records
    dicts = df[categorical].to_dict(orient='records')
    
    # Transform the dictionary using the DictVectorizer
    X_val = dv.transform(dicts)
    
    # Predict the durations using the model
    y_pred = model.predict(X_val)
    
    # Create a new DataFrame with ride_id and y_pred
    result_df = pd.DataFrame({
        'ride_id': df['ride_id'],
        'predicted_duration': y_pred
    })

    print('Mean of predicted duration: ', np.mean(y_pred))
    

    return result_df

def main(year, month, output_file):
    # Read and process the data
    df = read_data(year, month)
    
    # Load the model and vectorizer
    model, dv = load_model_and_vectorizer()
    
    # Score the data
    result_df = score_data(df, model, dv)
    
    # Save the result_df to a Parquet file
    result_df.to_parquet(output_file, engine='pyarrow', compression=None, index=False)

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    year = 2023
    month = 5
    output_file = "april_predictions.parquet"
    main(year, month, output_file)