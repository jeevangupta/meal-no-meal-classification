#!/usr/local/bin/python3
#command to run file : $ python3 ./main.py -a ./CGMData.csv -b InsulinData.csv
import sys
import os.path
import getopt
import numpy as np
from numpy.core.records import array
import pandas as pd
from datetime import datetime, timedelta

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle

def feature_extraction(data_df):
    try:
        pca = PCA(n_components=11)
        feature_matrix = pca.fit_transform(data_df)
        return feature_matrix
    except:
        print("\n *** Function (feature_extraction) failed *** ",sys.exc_info())


if __name__ == '__main__':
    try:
        timestamp = datetime.strftime(datetime.now(),'%Y-%m-%d')
        print("DATE : ",timestamp)
        print("Prediction of Meal and No Meal Data Process Starts.")

        test_file_name = "./data/test.csv"

        test_cgm_data = pd.read_csv(test_file_name)

        test_cgm_data_feature_matrix = feature_extraction(test_cgm_data)

        pkl_file_name = "./svm_model.pkl"
        # Load from file
        with open(pkl_file_name, 'rb') as file:
            svm_classifier = pickle.load(file)

        test_cgm_data_prediction = svm_classifier.predict(test_cgm_data_feature_matrix)
        test_cgm_data_prediction = test_cgm_data_prediction.astype(int)
        prediction_df = pd.DataFrame(test_cgm_data_prediction)
        #print(prediction_df)
  
        prediction_df.to_csv('./data/Result.csv', index=False)

    except:
        print("\n*** Extracting Meal and No Meal Data Process Starts. *** ", sys.exc_info())


