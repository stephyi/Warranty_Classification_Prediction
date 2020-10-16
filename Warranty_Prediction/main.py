import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import argparse

from model import Modelling
from pred_utilities import DealwithSample
from preprocessing import Preprocessor

class ModelTraining():
    def __init__(self, path='data\df_Clean.csv', label="Fraud", scaler= 'RobustScaler', sampler="SMOTE", model="RandomForest", validate=0.1, testdata=None):
        self.path = path
        self.scaler = scaler
        self.sampler = sampler
        self.label = label
        self.model = model
        self.validate= validate
        self.testdata = testdata 

    def train(self):
        print(path)
        if self.path == 'data\df_Clean.csv':
            data = pd.read_csv('data\df_Clean.csv')
        else:
            data = pd.read_csv(path)
        
        process = Preprocessor(data)
        data, columns = process._divide_data(data, self.label)
        categorical, numerical = process._classify_data(columns, data)
        transformed, y = process._preprocess_data(categorical, numerical, preprocessor=scaler)

        #dealing with our imbalanced data by oversampling the data set
        model = Modelling()
        x_train, x_test, y_train, y_test = model._splitdata(transformed, y, size=self.validate)
        X, y = DealwithSample(x_train, y_train, method=sampler)

        model = model.Prediction(X, x_test, y, y_test, method= self.model)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description="Base Models")
    parser.add_argument('--path', default='data\df_Clean.csv',
                    help='If there is anywhere you saved the train data, you can specify a link or path')
    parser.add_argument('--label', default='Fraud', help='If you wanted to specify the label of your data')
    parser.add_argument('--scaler',default='StandardScaler',
                    help='Method for scaling numerical data in dataset from(StandardScaler, RobustScaler)')
    parser.add_argument('--sampling', default='SMOTE',
                    help='Method to deal with imbalance data from this(SMOTE, ADASYN, RandomOverSampler) please spell as is')
    parser.add_argument('--model', default='RandomForest',
                    help='Model to use for training your data from this(RandomForest, DecisionTree, MultiPerceptronLayer, Classifier, )')
    parser.add_argument('--validationsize', default='0.1', help='Specify the test data size for this training must be a number')
    parser.add_argument('--testdata',default='None', help='Specify a link to your file here and we would run a prediction on your data' )
    args = parser.parse_args()
    path = str(args.path)
    label = str(args.label)
    scaler =str(args.scaler)
    sampler = str(args.sampling)
    model = str(args.model)
    validate = float(args.validationsize)
    testdata = str(args.testdata)
    model = ModelTraining(path, label, scaler, sampler, model, validate, testdata)
    model.train()
