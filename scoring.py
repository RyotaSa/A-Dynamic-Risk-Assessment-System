from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json


#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_path = os.path.join(config['prod_deployment_path']) 
model_path = os.path.join(config['output_model_path']) 


#################Function for model scoring
def score_model(test_data):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    y = test_data['exited']
    test_data.drop(['corporation', 'exited'], inplace=True, axis=1)

    with open(prod_path + '/trainedmodel.pkl', 'rb') as p:
        model = pickle.load(p)
    # model = pickle.load(model_path + '/trainedmodel.pkl', 'rb')
    pred = model.predict(test_data)
    f1 = metrics.f1_score(y, pred)

    with open(model_path + '/latestscore.txt', 'w') as f:
        f.write(str(f1))

    return f1


if __name__ == '__main__':
    test_data = pd.read_csv(os.path.join(config['test_data_path'],'testdata.csv'))
    score_model(test_data)