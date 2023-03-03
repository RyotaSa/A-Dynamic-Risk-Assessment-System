import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import glob
import os


#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']


#############Function for data ingestion
def merge_multiple_dataframe():
    #check for datasets, compile them together, and write to an output file
    all_files = glob.glob(os.path.join(input_folder_path, "*.csv"))

    name_list = []
    fi = []
    for file in all_files:
        name_list.append(os.path.basename(file))
        df = pd.read_csv(file)
        fi.append(df)
    df = pd.concat(fi, ignore_index=True)
    df.drop_duplicates(ignore_index=True, inplace=True)

    # Save dataframe to a csv file
    df.to_csv(output_folder_path + '/finaldata.csv', index=False)

    # Save file name to a text
    with open(output_folder_path + '/ingestedfiles.txt', 'w') as output:
        output.write(str(name_list))

    return df


if __name__ == '__main__':
    merge_multiple_dataframe()
