import pandas as pd 
import os 
import numpy as np 
import pickle
import logging 
from sklearn.ensemble import RandomForestClassifier
import yaml


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir,"model_training.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_param(params_path):
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s',params_path)
        return params
    except FileNotFoundError as e:
        logger.error("File not found at : %s",e)
        raise
    except yaml.YAMLError as e:
        logger.error('yamlerror : %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error: %s',e)
        raise

def load_data(file_path):
    """loading the tfidf data from a csv file """
    try:
        df = pd.read_csv(file_path)
        logger.debug("the file is loaded")
        return df
    except Exception as e:
        logger.error("the path of the file is may be not correct , this leads to : %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("the file may not be foud in the given path which leads to : %s",e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file: %s",e)
        raise
    
    
def train_model(x_train,y_train,model_param):
    """
    we are using random fores classfier model to classify the data in two classes: spam or ham
    :param x_train: Training features
    :param y_train: Training labels
    :param prams:Dictionary of hyperparameters of the randomforestcassifier model"""
    

    try:
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("The number of samlples in x_train and y_train must be same.")
        
        logger.debug("Initializing RandomForestClassifier model with the parametrs: %s",model_param)
        clf = RandomForestClassifier(n_estimators=model_param['n_estimators'],random_state=model_param['random_state'])
        logger.debug("Training started......")
        clf.fit(x_train,y_train)
        logger.debug("training completed....")
        return clf
    except ValueError as e:
        logger.error("value error during model training %s",e)
        raise
    except Exception as e:
        logger.error("the unexpected error occurs which leads to %s",e)
        raise
    
def save_model(model,file_path):
    """
    this function is used to save the model i a given file path 
    """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,"wb") as file:
            pickle.dump(model,file)
            
        logger.debug("Model is aved in the given file path : %s",file_path)
    except FileNotFoundError as e:
        logger.error("File path is not found : %s",e)
        raise
    except Exception as e:
        logger.error("File has some unexpected error leads to : %s",e)
        raise
    
def main():
    try:
        
        model_param = load_param(params_path='params.yaml')['model']
        train_data = load_data(os.path.join("data","processed","train_tfidf_processed.csv"))
        x_train = train_data.iloc[:,:-1].values
        y_train = train_data.iloc[:,-1].values
        
        clf = train_model(x_train,y_train,model_param)
        model_save_path = "models/model.pkl"
        save_model(clf,model_save_path)
    except Exception as e:
        logger.error("Failed to train the model due to sme unexpected errror : %s",e)
        raise
    
if __name__ == '__main__':
    main()

           
