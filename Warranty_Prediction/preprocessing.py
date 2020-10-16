import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class Preprocessor():
    def __init__(self,data):
        self.data = data
        self.column = data.columns
        self.numerical = [] 
        self.categorical = []
       
        
    

    def _divide_data(self,data, y=[]):
        """This divide the data into label and data so we can only preprocess our data and not label too, 
we would check if label was passed or not"""
        self.label = data.pop(y)
        columns = data.columns
        self.data = data
        self.column = columns
        return data, columns


    def _classify_data(self,columns,data):
        """This would help classify the data into numerical and categorical columns"""
        categorical = []
        numerical =[]
        categorical = data.select_dtypes(include=['object']).columns
        numerical = data.select_dtypes(include=['float64', 'int64']).columns
        self.numerical = numerical
        self.categorical= categorical
        return categorical, numerical


    def _preprocess_data(self,categorical,numerical):
        """This function helps to encode the categorical data using One Hot encoding
         and then Scale the numerical data using the well known Standard Scaler"""
        numeric_transformer = Pipeline([('scaler', RobustScaler())])
        categorical_transformer = Pipeline([('onehot', OneHotEncoder(sparse=False))])
        
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)])
        
        transformed  = Pipeline([('preprocess', preprocessor)])
        trans_fit  = transformed.fit_transform(self.data)
        label = self.label
        return trans_fit, label

    
        




