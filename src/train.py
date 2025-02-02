import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from mlflow.models import infer_signature
import os

from sklearn.model_selection import train_test_split,GridSearchCV
from urllib.parse import urlparse
import mlflow 

os.environ["MLFLOW_TRACKING_URI"]=""
os.environ["MLFLOW_TRACKING_USER_NAME"]=
os.environ["MLFLOW_TRACKING_PASSWORD"]=

def hyper_paramater_tunrning(X_train,Y_train,param_grid):
    rf=RandomForestClassifier()
    grid_serach=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_serach.fit(X_train,y_train)
    return  grid_serach

### Load the paramaters from params.yaml
params=yaml.safe_load(open("param.yaml"))["train"]

def train(data_path,model_path,random_state,n_estimators,max_depth):
     data=pd.read_csv(data_path)
     x=data.drop(columns=["Outcome"])
     y=data["Outcome"]
     mlflow.set_tracking_uri("")
     ## Start MLFLOW run 
     with mlflow.start_run():
          ## Split data into train and test 
          X_train,X_test,y_train,y_test=train_test_split(X,test_size=0.20)





