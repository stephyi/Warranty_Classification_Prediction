import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class Preprocessor():
    """
    This instantiate the class preprocessor which assist in preprocessing the data, it deals
    with outliers, encode our categorical variables and scale our
    """
    def __init__(self, data):
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

    def _preprocess_data(self, categorical, numerical, preprocessor="RobustScaler"):
        """This function helps to encode the categorical data using One Hot encoding
         and then Scale the numerical data using the well known Robust Scaler this help
         so that our data is not affected by outliers"""
        if preprocessor == "RobustScaler":   
             numeric_transformer = Pipeline([('scaler', RobustScaler())])
        else:
            numeric_transformer = Pipeline([('scaler', StandardScaler())])
        categorical_transformer = Pipeline([('onehot', OneHotEncoder(sparse=False))])
        
        preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical),
            ('cat', categorical_transformer, categorical)])
        
        transformed  = Pipeline([('preprocess', preprocessor)])
        trans_fit  = transformed.fit_transform(self.data)
        label = self.label
        return trans_fit, label

    
        


    
        




