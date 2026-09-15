import numpy as np 
import pandas as pd 
import os 
from sklearn.model_selection import train_test_split
import logging 


#ensure the log directories are availaible 
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)


#logging configuration
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_dir = os.path.join(log_dir,'data_ingestion.log')
file_handler = logging.FileHandler(log_file_dir)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url):
    """I just load the data from the csv file"""
    try:
        print("url checking")
        df = pd.read_csv(data_url)
        print("url is correct....")
        logger.debug('Data loaded from %s',data_url)
        return df
    except pd.errors.ParseError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error occured due to some unwanted circumstance: %s',e)
        raise
    
def preprocess_data(df):
    """preprocess the data"""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns = {'v1':'target','v2':'text'}, inplace = True)
        logger.debug('data preprocessing is completed')
        return df 
    except KeyError as e:
        logger.error('Missing olumn in the datarame: %s',e)
        raise
    except Exception as e:
        logger.error('unexpected error during preprocessing %s ',e)
        raise
    
    
def save_data(train_data,test_data,data_path):
    """Saving the training data and testiing data"""
    try:
        raw_data_path = os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index = False)  
        logger.debug('Train and test data is saved successfully: %s',raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while savig data: %s',e)
        raise
    
def main():
    try:
        test_size = 0.2
        data_path = "experiments\spam.csv" 
        print("data is loadng.....")
        df = pd.read_csv(data_path)
        print("reading of the data is done")
        print(df.shape)
        df = load_data(data_url=data_path)
        print("data is loading using load_data")
        final_df = preprocess_data(df=df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=2) 
        save_data(train_data,test_data,data_path='./data')
    except Exception as e:
        logger.error('Failed to complete ingestion process: %s',e)
        print(f'Error:{e}')






if __name__ == "__main__":
    main()