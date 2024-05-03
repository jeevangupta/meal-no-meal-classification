#!/usr/local/bin/python3
#command to run file : $ python3 ./main.py
import sys
import os.path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
from sklearn.tree import DecisionTreeClassifier


def SVM_classification(train_feature_matrix, train_feature_lable):
    try:
        svm_clf = svm.SVC()
        svm_clf.fit(train_feature_matrix, train_feature_lable)
        return svm_clf
    except:
        print("\n *** Function SVM_classification Failed ***",sys.exc_info() )

def DTC_classification(train_feature_matrix, train_feature_lable):
    try:
        dc_clf = DecisionTreeClassifier()
        dc_clf = dc_clf.fit(train_feature_matrix,train_feature_lable)
        return dc_clf
    except:
        print("\n *** Function DTC_classification Failed ***",sys.exc_info() )

def feature_extraction(data_df):
    try:
        pca = PCA(n_components=11)
        feature_matrix = pca.fit_transform(data_df)
        return feature_matrix
    except:
        print("\n *** Function (feature_extraction) failed *** ",sys.exc_info())


def get_meal_data_matrix(insulin_data,cgm_data):
    try:
        all_meal_time = insulin_data.query("Y != 0 and Y != 'NaN'")
        #print("\nall_meal_time \n",all_meal_time)

        #get all meal time
        final_meal_tm = []
        for i,r in all_meal_time.iterrows():
            tmp_tm = r['datetime'] 
            tmp_tm_2h = tmp_tm + timedelta(hours=2)
            #print("tmp_tm_2h:",tmp_tm_2h)

            mask1 = (all_meal_time['datetime'] > tmp_tm) & (all_meal_time['datetime'] < tmp_tm_2h)
            mask2 = (all_meal_time['datetime'] == tmp_tm_2h)
            if (all_meal_time.loc[mask1]).empty:
                final_meal_tm.append(tmp_tm)
            if not (all_meal_time.loc[mask2]).empty:
                final_meal_tm.append(tmp_tm_2h)

        #print("\n** Final TM:", final_meal_tm)

        #get all cgm data wrt to meal time tm
        m = []
        for dt in final_meal_tm:
            start_mt = dt - timedelta(minutes=30)
            end_mt = dt + timedelta(hours=2)
            #print("\nstart_mt: ",start_mt ," end_mt:", end_mt)
            cgm_meal_data = cgm_data[(cgm_data['datetime']>=start_mt) & (cgm_data['datetime']<=end_mt)]
            #print(cgm_meal_data)
            #print(len(cgm_meal_data.index))
            r = cgm_meal_data['CGM'].to_numpy()
            nan_count = np.count_nonzero(np.isnan(r))
            if nan_count < 15 and len(r) >=30:
                r = np.nan_to_num(r)
                if len(r)>30:
                    r = r[0:30]
                m.append(r)

        
        meal_data_matrix = np.array(m,dtype=object)
        
        return meal_data_matrix,all_meal_time,final_meal_tm
    except:
        print("\n*** Function (get_meal_data_matrix) Failed **** ",sys.exc_info())


def get_no_meal_data_matrix(cgm_data,final_meal_tm,all_meal_time):
    try:
        #get all post meal time
        post_meal_tm = []
        for i in final_meal_tm:
            tmp_s_pmt = i + timedelta(hours=2)
            tmp_s_pmt_2h = tmp_s_pmt + timedelta(hours=2)
            #print("tmp_s_pmt: ",tmp_s_pmt, "tmp_s_pmt_2h: ",tmp_s_pmt_2h)
            n=0
            while n < 6:
                n = n+1
                mask1 = (all_meal_time['datetime'] > tmp_s_pmt) & (all_meal_time['datetime'] <=tmp_s_pmt_2h)
                if (all_meal_time.loc[mask1]).empty:
                    post_meal_tm.append(tmp_s_pmt)
                    tmp_s_pmt = tmp_s_pmt_2h
                    tmp_s_pmt_2h = tmp_s_pmt + timedelta(hours=2)

        #print("\n** post_meal_tm: ", post_meal_tm)

        m = []
        for dt in post_meal_tm:
            start_pmt = dt
            end_pmt = dt + timedelta(hours=2)
            #print("\nstart_pmt: ",start_pmt ," end_pmt:", end_pmt)
            cgm_no_meal_data = cgm_data[(cgm_data['datetime']>start_pmt) & (cgm_data['datetime']<=end_pmt)]
            size = len(cgm_no_meal_data.index)
            #print(cgm_no_meal_data)
            r = cgm_no_meal_data['CGM'].to_numpy()
            nan_count = np.count_nonzero(np.isnan(r))
            if nan_count < 15 and size >= 24:
                r = np.nan_to_num(r)
                if len(r)>24:
                    r = r[0:24]
                m.append(r)
            
        no_meal_data_matrix = np.array(m)
        
        return no_meal_data_matrix
    except:
        print("\n*** Function (get_meal_data_matrix) Failed **** ",sys.exc_info())


if __name__ == '__main__':
    try:
        timestamp = datetime.strftime(datetime.now(),'%Y-%m-%d')
        print("DATE : ",timestamp)
        print("Extracting Meal and No Meal Data Process Starts.")

        if(os.path.isfile("./data/CGMData.csv")):
            cgm_data_file = "./data/CGMData.csv"
        elif(os.path.isfile("./data/CGM_patient2.csv")):
            cgm_data_file = "./data/CGM_patient2.csv"
        
        if(os.path.isfile("./InsulinData.csv")):
            insulin_data_file = "./InsulinData.csv"
        elif(os.path.isfile("./Insulin_patient2.csv")):
            insulin_data_file = "./Insulin_patient2.csv"


        cgm_data = pd.read_csv(cgm_data_file, usecols=['Date','Time','Sensor Glucose (mg/dL)','ISIG Value'])
        insulin_data = pd.read_csv(insulin_data_file, usecols=['Date','Time','BWZ Carb Input (grams)'])


        insulin_data['datetime'] = pd.to_datetime(insulin_data['Date']+" "+insulin_data['Time'],infer_datetime_format=True)
        insulin_data.rename(columns = {'BWZ Carb Input (grams)':'Y'}, inplace = True)
        #insulin_data1 = insulin_data[insulin_data['datetime']>='2018-01-01']
        #print("\n*** insulin_data1 *** \n",insulin_data1)


        cgm_data['datetime'] = pd.to_datetime(cgm_data['Date']+" "+cgm_data['Time'],infer_datetime_format=True)
        cgm_data.rename(columns = {'Sensor Glucose (mg/dL)':'CGM'}, inplace = True)
        #cgm_data1 = cgm_data[cgm_data['datetime']>='2018-01-01']
        #print("\n*** gm_data1 ****\n",cgm_data1)

        meal_data_matrix,all_meal_time,final_meal_tm = get_meal_data_matrix(insulin_data,cgm_data)
        #print("\n*** meal_data_matrix ***\n ",meal_data_matrix)
        meal_feature_matrix = feature_extraction(meal_data_matrix)
        #print("\n*** meal_feature_matrix ***\n ",meal_feature_matrix)
        
        no_meal_data_matrix = get_no_meal_data_matrix(cgm_data,final_meal_tm,all_meal_time)
        #print("\n*** no_meal_data_matrix *** \n",no_meal_data_matrix)
        no_meal_feature_matrix = feature_extraction(no_meal_data_matrix)
        #print("\n*** no_meal_feature_matrix ***\n ",no_meal_feature_matrix)

        if len(meal_feature_matrix) > 0 and len(no_meal_feature_matrix)>0:
            feature_matrix = np.concatenate((meal_feature_matrix, no_meal_feature_matrix))
            #print("\n*** feature_matrix *** \n",feature_matrix)
        
        if len(feature_matrix) > 0:
            ml_n = meal_feature_matrix.shape[0]
            #ml_lable = np.ones(shape=(ml_n,1))
            ml_lable = np.ones(ml_n)

            no_ml_n = no_meal_feature_matrix.shape[0]
            #no_ml_lable = np.zeros(shape=(no_ml_n,1))
            no_ml_lable = np.zeros(no_ml_n)
        
            feature_lable = np.concatenate((ml_lable, no_ml_lable))
            #print("\n*** feature_lable *** \n",feature_lable)

            train_feature_matrix, test_feature_matrix, train_feature_lable, test_feature_lable = train_test_split(feature_matrix, feature_lable, test_size=0.2,random_state=68)

            #clf = SVM_classification(train_feature_matrix, train_feature_lable)
            clf = DTC_classification(train_feature_matrix, train_feature_lable)
            test_feature_matrix_prediction = clf.predict(test_feature_matrix)
            
            #test_feature_matrix_prediction = test_feature_matrix_prediction.astype(int)
            #print(test_feature_matrix_prediction)
            print("\nScore:",clf.score(test_feature_matrix,test_feature_lable))
            print("\nAccuracy:",metrics.accuracy_score(test_feature_lable, test_feature_matrix_prediction))
            print("\nPrecision:",metrics.precision_score(test_feature_lable, test_feature_matrix_prediction))
            print("\nRecall:",metrics.recall_score(test_feature_lable, test_feature_matrix_prediction))
            print("\nF1 Score:",metrics.f1_score(test_feature_lable, test_feature_matrix_prediction))

            # Save to file in the current working directory
            pkl_file_name = "./data/svm_model.pkl"
            with open(pkl_file_name, 'wb') as file:
                pickle.dump(clf, file)


        #prediction_df = pd.DataFrame(test_feature_matrix_prediction)
        #print(prediction_df)
        #prediction_df.to_csv('./Result.csv', index=False, header=False)

        #test_df = pd.DataFrame(meal_data_matrix)
        #test_df.to_csv('./test.csv', index=False, header=False)

    except:
        print("\n*** Extracting Meal and No Meal Data Process Starts. *** ", sys.exc_info())


