"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.

"""

# Helper Dependencies
import json
import pickle
import collections
import numpy as np
import pandas as pd

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])
    test_data = feature_vector_df
    # ------------------------------------------------------------------------
    #### Load Data
    # Fetch data and preprocess for modeling
    train_data = pd.read_csv('utils/data/Train.csv')
    riders = pd.read_csv('utils/data/Riders.csv')
    # Rename columns
    test_data.columns = [column.replace("Of","of") for column in test_data.columns]
    test_data.columns = [column.replace("_"," ") for column in test_data.columns]
    test_data.columns = [column.replace("(Mo = 1)"," ") for column in test_data.columns]
    test_data = test_data.rename(columns=lambda x: x.strip())

    #### Data Cleaning
    # Drop 'Precipitation in millimeters' column
    test_data.drop('Precipitation in millimeters',axis=1,inplace=True)
    # Create 24h time bins for when orders were placed
    train_data['Placement - Time(bins)'] = pd.to_datetime(pd.to_datetime(train_data['Placement - Time']).dt.strftime('%H:%M:%S')).dt.strftime('%H')
    # Impute temperature for missing values
    test_data['Temperature'] = test_data['Temperature'].fillna(round(train_data.groupby('Placement - Time(bins)')['Temperature'].transform('mean'),1))

    #### Feature Engineering
    # Function to calculate time in seconds (from midnight)
    def time_in_seconds(df,column):
        df[column] = pd.to_datetime(df[column])
        return (df[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')
    test_data['Pickup - Time'] = time_in_seconds(test_data,'Pickup - Time')

    #### Encoding Categorical Data
    # Change 'Platform Type' data type
    model_features = ['Pickup - Day of Month', 'Pickup - Weekday', 'Pickup - Time',
       'Distance (KM)', 'Temperature', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long', 'No of Orders',
       'Age', 'Average Rating', 'No of Ratings']
    # Drop unnecessary columns
    X_test = test_data[model_features]
    # ------------------------------------------------------------------------

    return X_test

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
