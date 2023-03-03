import sys
import os
import json
import glob
import pandas as pd

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import apicalls

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
input_folder_path = config['input_folder_path']

##################Check and read new data
#first, read ingestedfiles.txt
input_files_path = glob.glob(input_folder_path + '/*.csv')
input_files = [os.path.split(f)[1] for f in input_files_path]

with open(os.path.join(os.getcwd(), config['prod_deployment_path'], 'ingestedfiles.txt'), 'r') as f:
    ingestedfiles = f.readlines()
ingestedfiles = [ingestedfile.replace('\n','') for ingestedfile in ingestedfiles]
##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
file_list = list(set(input_files) - set(ingestedfiles))

if len(file_list) == 0:
    sys.exit()
##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
with open(os.path.join(config['prod_deployment_path'], 'latestscore.txt'), 'r') as f:
    f1 = f.read()

df = ingestion.merge_multiple_dataframe()
new_f1 = scoring.score_model(df)
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if float(new_f1) > float(f1):
    sys.exit()

# Retraining
training.train_model()
##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
deployment.store_model_into_pickle()
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
summary = diagnostics.dataframe_summary()
print('Dataframe summary statistics:' + str(summary))

missing_data_ratio = diagnostics.missing_data(df)
print('Missing data ratio:' + str(missing_data_ratio))

executing_time = diagnostics.execution_time()
print('Executing time:' + str(executing_time))

package_list = diagnostics.outdated_packages_list()
print(package_list)


confusion_matrix_filepath= '/confusionmatrix2.png'
reporting.score_model(confusion_matrix_filepath)

api_filename = 'apireturns2.txt'
apicalls.api_calls(api_filename)







