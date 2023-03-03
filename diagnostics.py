import pandas as pd
import numpy as np
import timeit
import os
import json
import time
import pickle
import subprocess


##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])


##################Function to get model predictions
def model_predictions(df):
    #read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path + '/trainedmodel.pkl', 'rb') as p:
        model = pickle.load(p)
    pred = model.predict(df)
    return pred

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    df.drop(['corporation', 'exited'], inplace=True, axis=1)
    # statistics = []
    statistics = {}
    for i in df.columns.tolist():
        mean = np.mean(df[i])
        median = np.median(df[i])
        std = np.std(df[i])
        # statistics.append([mean, median, std])
        statistics[i] = [('Mean', mean), ('Median', median), ('Standard deviation', std)]
        # statistics[i] = [mean, median, std]

    return statistics

# Calculate what percent of each column consists of NA values
def missing_data(df):
    value_list = []
    length_df = len(df)
    for i in df.columns.tolist():
        nan_count = df[i].isna().sum()
        nan_ratio = nan_count / length_df * 100
        value_list.append(nan_ratio)

    return value_list

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    time_list = []
    start_1 = time.time()
    os.system('python3 ingestion.py')
    end_1 = time.time() - start_1
    start_2 = time.time()
    os.system('python3 training.py')
    end_2 = time.time() - start_2
    time_list.append(end_1)
    time_list.append(end_2)

    return time_list

##################Function to check dependencies
def outdated_packages_list():
    # python module name
    args1 = ['pip', 'list']
    res1 = subprocess.check_output(args1)
    
    # currently installed version of Python module
    args2 = ['pip', 'freeze']
    res2 = subprocess.check_output(args2)

    # most recent available version of Python module
    args3 = ['pip', 'list', '-o']
    res3 = subprocess.check_output(args3)

    # package = {'module name':res1, 'Installed python module':res2, 'Available version of python module':res3}
    package = {'module name':res1.decode('utf-8'), 'Installed python module':res2.decode('utf-8'), 'Available version of python module':res3.decode('utf-8')}

    return package


if __name__ == '__main__':
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    df.drop(['corporation', 'exited'], inplace=True, axis=1)

    # print(model_predictions(df))
    # print(dataframe_summary(df))
    # print(missing_data(df))
    # print(execution_time())
    # print(outdated_packages_list())

    model_predictions(df)
    dataframe_summary()
    missing_data(df)
    execution_time()
    outdated_packages_list()





    
