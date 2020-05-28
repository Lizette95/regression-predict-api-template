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
import pygeohash as gh
import lightgbm as lgbm
from scipy.stats import boxcox, zscore
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

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
# Calculate delivery rate for riders
train_data['Delivery Rate'] = train_data['No of Orders']/train_data['Age']
# Calculate scaled rating for riders
train_data['Scaled Rating'] = train_data['Average Rating']*(train_data['No of Ratings']/train_data['No of Ratings'].sum())
# Convert pickup day of month feature to cyclical format
train_data['Pickup - Day of Month (sin)'] = np.sin((train_data['Pickup - Day of Month'])*(2.*np.pi/31))
train_data['Pickup - Day of Month (cos)'] = np.cos((train_data['Pickup - Day of Month'])*(2.*np.pi/31))
# Convert pickup weekday feature to cyclical format
train_data['Pickup - Weekday (sin)'] = np.sin(train_data['Pickup - Weekday']*(2.*np.pi/7))
train_data['Pickup - Weekday (cos)'] = np.cos(train_data['Pickup - Weekday']*(2.*np.pi/7))
# Function to calculate time in seconds (from midnight)
def time_in_seconds(df,column):
    df[column] = pd.to_datetime(df[column])
    return (df[column] - pd.to_datetime(pd.to_datetime('today').date())).astype('timedelta64[s]')
# Convert pickup time feature to cyclical format
train_data['Pickup - Time'] = time_in_seconds(train_data,'Pickup - Time')
train_data['Pickup - Time (sin)'] = np.sin(train_data['Pickup - Time']*(2.*np.pi/86400))
train_data['Pickup - Time (cos)'] = np.cos(train_data['Pickup - Time']*(2.*np.pi/86400))
# Convert coordinates to geohash encoding
train_data['Pickup (geohash)'] = train_data.apply(lambda x: gh.encode(x['Pickup Lat'], x['Pickup Long'], precision=5), axis=1)
train_data['Destination (geohash)'] = train_data.apply(lambda x: gh.encode(x['Destination Lat'], x['Destination Long'], precision=5), axis=1)
# Encode geohash labels
encoder_dict = collections.defaultdict(LabelEncoder)
labeled_df = train_data[['Pickup (geohash)','Destination (geohash)']].apply(lambda x: encoder_dict[x.name].fit_transform(x))
train_data['Pickup (label)'] = labeled_df['Pickup (geohash)']
train_data['Destination (label)'] = labeled_df['Destination (geohash)']
# Transform target variable and remove high Z-scores
train_data_t = train_data.copy()
train_data_t['transformed'] = boxcox(train_data['Time from Pickup to Arrival'])[0]
train_data_t['zscore'] = zscore(train_data_t['transformed'])
train_data_t = train_data_t[train_data_t['zscore'].abs() < 3]
train_data_t = train_data_t.drop('zscore', axis=1)

#### Encoding Categorical Data
# Change 'Platform Type' data type
train_data_t['Platform Type'] = train_data_t['Platform Type'].astype(str)
model_features = ['Platform Type','Personal or Business',
       'Pickup - Day of Month', 'Pickup - Weekday', 'Pickup - Time',
       'Distance (KM)', 'Temperature', 'Pickup Lat', 'Pickup Long',
       'Destination Lat', 'Destination Long', 'No of Orders',
       'Age', 'Average Rating', 'No of Ratings', 'Delivery Rate', 'Scaled Rating',
       'Pickup - Day of Month (sin)', 'Pickup - Day of Month (cos)',
       'Pickup - Weekday (sin)', 'Pickup - Weekday (cos)', 'Pickup - Time (sin)',
       'Pickup - Time (cos)', 'Pickup (label)', 'Destination (label)',
       'Time from Pickup to Arrival']
# Drop unnecessary columns
train = train_data_t[model_features]
# Rearrange columns
train['Time from Pickup to Arrival'] = train.pop('Time from Pickup to Arrival')
# Create matrix of features
X_train = train.iloc[:, :-1].values
y_train = train.iloc[:, -1].values
# Encode categorical data
label = LabelEncoder()
X_train[:,1] = label.fit_transform(X_train[:,1]) #Customer Type
# ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough') #Platform Type
# X_train = np.array(ct.fit_transform(X_train))
# # Drop last Platform Type dummy variable
# X_train = np.delete(X_train,3,axis=1)

#### Fit Model
lgbm_model = lgbm.LGBMRegressor()
lgbm_model.fit(X_train,y_train)

# Pickle Model for API
save_path = '../assets/trained-models/test.pkl'
print (f"Training completed. Saving model to: {save_path}")
with open(save_path,'wb') as file:
    pickle.dump(lgbm_model, file)
