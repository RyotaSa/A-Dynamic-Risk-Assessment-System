import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

from diagnostics import model_predictions

from sklearn.metrics import confusion_matrix


###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 


##############Function for reporting
def score_model(confusion_matrix_filepath):
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    df = pd.read_csv(test_data_path + '/testdata.csv')
    y = df['exited']
    df.drop(['corporation', 'exited'], inplace=True, axis=1)

    pred = model_predictions(df)
    matrix_plot = ConfusionMatrixDisplay(confusion_matrix(y, pred))
    matrix_plot.plot()
    plt.savefig(output_model_path + confusion_matrix_filepath)


if __name__ == '__main__':
    confusion_matrix_filepath = '/confusionmatrix.png'
    score_model(confusion_matrix_filepath)
