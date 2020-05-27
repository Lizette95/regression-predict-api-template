"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

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
import pygeohash as gh
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


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
    train_data = pd.read_csv('https://raw.githubusercontent.com/Lizette95/regression-predict-api-template/master/utils/data/Train.csv')
    riders = pd.read_csv('https://raw.githubusercontent.com/Lizette95/regression-predict-api-template/master/utils/data/Riders.csv')
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
    # Calculate delivery rate for riders
    test_data['Delivery Rate'] = test_data['No of Orders']/test_data['Age']
    # Calculate scaled rating for riders
    test_data['Scaled Rating'] = test_data['Average Rating']*(test_data['No of Ratings']/test_data['No of Ratings'].sum())

    # Convert pickup day of month feature to cyclical format
    test_data['Pickup - Day of Month (sin)'] = np.sin((test_data['Pickup - Day of Month'])*(2.*np.pi/31))
    test_data['Pickup - Day of Month (cos)'] = np.cos((test_data['Pickup - Day of Month'])*(2.*np.pi/31))
    # Convert pickup weekday feature to cyclical format
    test_data['Pickup - Weekday (sin)'] = np.sin(test_data['Pickup - Weekday']*(2.*np.pi/7))
    test_data['Pickup - Weekday (cos)'] = np.cos(test_data['Pickup - Weekday']*(2.*np.pi/7))
    # Function to calculate time in seconds (from midnight)
    def time_in_seconds(df,column):
        df[column] = pd.to_datetime(df[column])
        return (df[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')
    # Convert pickup time feature to cyclical format
    test_data['Pickup - Time'] = time_in_seconds(test_data,'Pickup - Time')
    test_data['Pickup - Time (sin)'] = np.sin(test_data['Pickup - Time']*(2.*np.pi/86400))
    test_data['Pickup - Time (cos)'] = np.cos(test_data['Pickup - Time']*(2.*np.pi/86400))
    # Convert coordinates to geohash encoding
    train_data['Pickup (geohash)'] = train_data.apply(lambda x: gh.encode(x['Pickup Lat'], x['Pickup Long'], precision=5), axis=1)
    train_data['Destination (geohash)'] = train_data.apply(lambda x: gh.encode(x['Destination Lat'], x['Destination Long'], precision=5), axis=1)
    test_data['Pickup (geohash)'] = test_data.apply(lambda x: gh.encode(x['Pickup Lat'], x['Pickup Long'], precision=5), axis=1)
    test_data['Destination (geohash)'] = test_data.apply(lambda x: gh.encode(x['Destination Lat'], x['Destination Long'], precision=5), axis=1)
    # Encode geohash labels in training set
    encoder_dict = collections.defaultdict(LabelEncoder)
    labeled_df = train_data[['Pickup (geohash)','Destination (geohash)']].apply(lambda x: encoder_dict[x.name].fit_transform(x))
    train_data['Pickup (label)'] = labeled_df['Pickup (geohash)']
    train_data['Destination (label)'] = labeled_df['Destination (geohash)']
    # Create label dictionaries from training set
    train_data[['Pickup (label)','Destination (label)']] += 1
    pickup_labels = pd.Series(train_data['Pickup (label)'].values,index=train_data['Pickup (geohash)']).to_dict()
    destination_labels = pd.Series(train_data['Destination (label)'].values,index=train_data['Destination (geohash)']).to_dict()
    # Encode geohash labels in testing set
    test_data['Pickup (label)'] = test_data['Pickup (geohash)'].apply(lambda i: pickup_labels[i] if i in pickup_labels.keys() else 0)
    test_data['Destination (label)'] = test_data['Destination (geohash)'].apply(lambda i: destination_labels[i] if i in destination_labels.keys() else 0)

    #### Encoding Categorical Data
    # Change 'Platform Type' data type
    test_data['Platform Type'] = test_data['Platform Type'].astype(str)
    model_features = ['Platform Type','Personal or Business',
           'Pickup - Day of Month', 'Pickup - Weekday', 'Pickup - Time',
           'Distance (KM)', 'Temperature', 'Pickup Lat', 'Pickup Long',
           'Destination Lat', 'Destination Long', 'No of Orders',
           'Age', 'Average Rating', 'No of Ratings', 'Delivery Rate', 'Scaled Rating',
           'Pickup - Day of Month (sin)', 'Pickup - Day of Month (cos)',
           'Pickup - Weekday (sin)', 'Pickup - Weekday (cos)', 'Pickup - Time (sin)',
           'Pickup - Time (cos)', 'Pickup (label)', 'Destination (label)']
    # Drop unnecessary columns
    test = test_data[model_features]

    # Create matrix of features
    X = test.iloc[:,:].values

    # Encode categorical data
    label = LabelEncoder()
    X[:,1] = label.fit_transform(X[:,1]) #Customer Type
    # ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #Platform Type
    # X = np.array(ct.fit_transform(X))
    # Drop last Platform Type dummy variable
    # X = np.delete(X,3,axis=1)
    print(str(X.shape))
    columns = ['Platform Type','Personal or Business', 'Pickup - Day of Month',
       'Pickup - Weekday', 'Pickup - Time', 'Distance (KM)', 'Temperature',
       'Pickup Lat', 'Pickup Long', 'Destination Lat', 'Destination Long',
       'No of Orders', 'Age', 'Average Rating', 'No of Ratings',
       'Delivery Rate', 'Scaled Rating', 'Pickup - Day of Month (sin)',
       'Pickup - Day of Month (cos)', 'Pickup - Weekday (sin)',
       'Pickup - Weekday (cos)', 'Pickup - Time (sin)', 'Pickup - Time (cos)',
       'Pickup (label)', 'Destination (label)']
    X_test = pd.DataFrame(X,columns=columns)
    X_test = X_test.apply(pd.to_numeric)
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
