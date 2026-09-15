import numpy as np 
import os 
import pandas as pd 
import pickle 
import logging
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,roc_auc_score,precision_score


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger("model_evaluation.log")
logger.setLevel("DEBUG")

console_logger = logging.StreamHandler()
console_logger.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_logger = logging.FileHandler(log_file_path)
file_logger.setLevel('DEBUG')

formatter = logging.Formatter(" %(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_logger.setFormatter(formatter)
file_logger.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_logger)


def load_model(file_path):
    """loading the model"""
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
        
        logger.debug("model is loaded ... ")
        return model
    except Exception as e:
        logger.error("model is not loaded due to some unexpected error: %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("File path is not found: %s",e)
        raise
    
def load_data(file_path):
    """load the data from the csv file"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Data is loaded sucessfully")
        return df 
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the csv file : %s",e)
        raise
    except FileNotFoundError as e:
        logger.error("File is not found in the given path : %s",e)
        raise
    except Exception as e:
        logger.error("some error occured dur to unexpected error : %s",e)
        raise
    
    
def evaluate_model(clf,x_test,y_test):
    """ 
    Find out the metrics to show how much my model is sucessfulll
    """
    try:
        y_pred = clf.predict(x_test)
        y_pred_probab = clf.predict_proba(x_test)[:,1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred)
        
        metrics_dict = {
            'accuracy' : accuracy,
            'precision' : precision,
            'recall' : recall,
            'auc' : auc
        }
        
        logger.debug('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logger.error('Error during model evaluation: %s',e)
        raise
    
def save_metrics(metrics,file_path):
    """saving the metrics of the model in a separated file """
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug("metrics saved to %s",file_path)
    except Exception as e:
        logger.error('Error ocurred while saving the metrics: %s',e)
        raise
    except FileNotFoundError as e:
        logger.error('File is not foud at the given path %s',e)
        raise
    
    
def main():
    try:
        clf = load_model("models/model.pkl")
        test_data = load_data("data/processed/test_tfidf_processed.csv")
        
        x_test = test_data.iloc[:,:-1].values
        y_test = test_data.iloc[:,-1].values
        metrics = evaluate_model(clf,x_test,y_test)
        metrics_file_path = os.path.join("data","reports","metrics.json")
        save_metrics(metrics,metrics_file_path)
        
    except Exception as e:
        logger.error("Failed to complete the model evaluation process: %s",e)
        print(f"Error:{e}")
        
if __name__ == "__main__":
    main()
            
        
        