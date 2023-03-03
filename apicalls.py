import requests
import os
import json

with open('config.json','r') as f:
    config = json.load(f) 

#Specify a URL that resolves to your workspace
# URL = "http://127.0.0.1:8000/"
# # URL = "http://localhost:8000/"


def api_calls(api_filename):
    # URL = "http://127.0.0.1:8000/"
    URL = "http://localhost:8000/"
    model_path = os.path.join(config['output_model_path'])

    #Call each API endpoint and store the responses
    response1 = requests.post(URL + '/prediction?data=testdata/testdata.csv').content
    response2 = requests.get(url=URL + 'scoring').content
    response3 = requests.get(url=URL + 'summarystats').content
    response4 = requests.get(url=URL + 'diagnostics').content

    #combine all API responses
    responses = {'predictions': response1, 'scoring': response2, 'summarystats': response3, 'diagnostics': response4}

    #write the responses to your workspace
    with open(os.path.join(config['output_model_path'], api_filename), 'w') as f:
        f.write(str(responses))


if __name__ == "__main__":
    api_filename = 'apireturns2.txt'
    api_calls(api_filename)
