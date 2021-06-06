
import numpy as np
import pandas as pd
import os
import pickle
from google.cloud import storage


## Global model variable
model = None


# Download model file from cloud storage bucket
def download_model_file():

    from google.cloud import storage

    # Model Bucket details
    BUCKET_NAME        = "iris_model_functions"
    PROJECT_ID         = "bugrahansntrk"
    GCS_MODEL_FILE     = "iris_model_jan_2020_v1.pkl"

    # Initialise a client
    client   = storage.Client(PROJECT_ID)
    
    # Create a bucket object for our bucket
    bucket   = client.get_bucket(BUCKET_NAME)
    
    # Create a blob object from the filepath
    blob     = bucket.blob(GCS_MODEL_FILE)
    
    folder = '/tmp/'
    if not os.path.exists(folder):
      os.makedirs(folder)
    # Download the file to a destination
    blob.download_to_filename(folder + "local_model.pkl")



def predictw_iris(request):
  global model

  if not model:
    download_model_file()
    model = pickle.load(open("/tmp/local_model.pkl", 'rb'))
    params = request.get_json()

  if (params is not None) and ('features' in params):
    # Run a test prediction
    pred_species  = model.predict(np.array([params['features']]))
    return pred_species[0]
  else:
    return "There is nothing to predict."
