from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
# import create_prediction_model
# import diagnosis 
# import predict_exited_from_saved_model
import json
import os

from diagnostics import model_predictions
from diagnostics import dataframe_summary
from diagnostics import missing_data
from diagnostics import execution_time
from diagnostics import outdated_packages_list
from scoring import score_model


######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 

prediction_model = None


#######################Prediction Endpoint
@app.route("/prediction", methods=['POST','OPTIONS'])
def prediction():        
    #call the prediction function you created in Step 3
    datapath = request.args.get("data")
    df = pd.read_csv(datapath)
    df.drop(['corporation', 'exited'], inplace=True, axis=1)

    pred = model_predictions(df)
    return str(pred)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def scoring():        
    #check the score of the deployed model
    test_data_path = os.path.join(config['test_data_path'])
    test_data = pd.read_csv(os.path.join(os.getcwd(), test_data_path, 'testdata.csv'))
    f1 = score_model(test_data)
    return str(f1) #add return value (a single F1 score number)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def summarystats():
    #check means, medians, and modes for each column
    statistics = dataframe_summary()

    return statistics #return a list of all calculated summary statistics

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnostics():
    #check timing and percent NA values
    df = pd.read_csv(dataset_csv_path + '/finaldata.csv')
    df.drop(['corporation', 'exited'], inplace=True, axis=1)

    value_list = missing_data(df)
    exe_time = execution_time()
    package_list = outdated_packages_list()
    diagnosis = {'missing data': value_list, 'execution time': exe_time, 'package list': package_list}
    return diagnosis #add return value for all diagnostics

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
