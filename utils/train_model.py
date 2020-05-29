"""
    Simple file to create a Sklearn model for deployment in our API

    Author: Explore Data Science Academy

    Description: This script is responsible for training a simple linear
    regression model which is used within the API for initial demonstration
    purposes.

"""

# Dependencies
import pickle
import collections
import numpy as np
import pandas as pd
import lightgbm as lgbm
from scipy.stats import boxcox, zscore

#### Load Data
# Fetch training data and preprocess for modeling
train_df = pd.read_csv('data/Train.csv')
riders = pd.read_csv('data/Riders.csv')
# Merge datasets
train_data = pd.merge(train_df,riders,on='Rider Id',how='left')
# Rename columns
train_data.columns = [column.replace("Of","of") for column in train_data.columns]
train_data.columns = [column.replace("_"," ") for column in train_data.columns]
train_data.columns = [column.replace("(Mo = 1)"," ") for column in train_data.columns]
train_data = train_data.rename(columns=lambda x: x.strip())

#### Data Cleaning
# Drop 'Precipitation in millimeters' column
train_data.drop('Precipitation in millimeters',axis=1,inplace=True)
# Create 24h time bins for when orders were placed
train_data['Placement - Time(bins)'] = pd.to_datetime(pd.to_datetime(train_data['Placement - Time']).dt.strftime('%H:%M:%S')).dt.strftime('%H')
# Impute temperature for missing values
train_data['Temperature'] = train_data['Temperature'].fillna(round(train_data.groupby('Placement - Time(bins)')['Temperature'].transform('mean'),1))
#Identify and remove outliers in target variable
train_data['Outlier'] = (train_data['Distance (KM)']*30) > train_data['Time from Pickup to Arrival']
train_data = train_data[train_data['Outlier'] == False]
train_data = train_data.drop(['Outlier'],axis=1)

#### Feature Engineering
# Function to calculate time in seconds (from midnight)
def time_in_seconds(df,column):
    df[column] = pd.to_datetime(df[column])
    return (df[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')
train_data['Pickup - Time'] = time_in_seconds(train_data,'Pickup - Time')
# Transform target variable and remove high Z-scores
train_data_t = train_data.copy()
train_data_t['transformed'] = boxcox(train_data['Time from Pickup to Arrival'])[0]
train_data_t['zscore'] = zscore(train_data_t['transformed'])
train_data_t = train_data_t[train_data_t['zscore'].abs() < 3]
train_data_t = train_data_t.drop('zscore', axis=1)

#### Feature Selection
model_features = ['Pickup - Day of Month', 'Pickup - Weekday', 'Pickup - Time',
       'Distance (KM)', 'Temperature', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long', 'No of Orders',
       'Age', 'Average Rating', 'No of Ratings','Time from Pickup to Arrival']
# Drop unnecessary columns
train = train_data_t[model_features]
# Rearrange columns
train['Time from Pickup to Arrival'] = train.pop('Time from Pickup to Arrival')
# Create matrix of features
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values

#### Fit Model
lgbm_model = lgbm.LGBMRegressor(learning_rate=0.1, num_leaves=50, objective='regression')
lgbm_model.fit(X_train,y_train)

# Pickle Model for API
save_path = '../assets/trained-models/Team19_JHB_model.pkl'
print (f"Training completed. Saving model to: {save_path}")
with open(save_path,'wb') as file:
    pickle.dump(lgbm_model, file)
