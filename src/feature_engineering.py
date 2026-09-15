import os 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
import logging


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG') 

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_path):
    """load the data first"""
    try:
        df = pd.read_csv(data_path)
        if df.isna().sum().sum() > 0:
            df.fillna("",inplace=True)
        else:
            pass
        logger.debug('Data loaded and NaNs filled from %s',data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the csv file: %s',e)
        raise
    except Exception as e:
        logger.error("Unexpected error occurred while loading the data : %s",e)
        raise
    
def apply_tfidf(train_data,test_data,max_features):
    """apply Tfidf to the data"""
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)
        x_train = train_data['text'].values
        x_test = test_data['text'].values
        y_train = train_data['target'].values
        y_test = test_data['target'].values    
        
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)
        
        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df['label'] = y_test
        
        logger.debug('Bag of words applied and data transformed ')
        return train_df,test_df
    except Exception as e:
        logger.error('Error during Bag of words transformation: %s',e)
        raise
def save_data(df,file_path):
    """save the dataframe into a specific csv file"""
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug("The dataframe is saved in the csv file")
    except Exception as e:
        logger.error("file is may be curropted: %s",e)
        raise
    
def main():
    """we need to call every function using main function"""
    try:
        max_features = 50
        train_data = load_data('data/interim/train_processed.csv')
        test_data = load_data('data/interim/test_processed.csv')
        
        
        train_df,test_df= apply_tfidf(train_data,test_data,max_features=max_features)
        
        save_data(train_df,file_path=os.path.join("data/processed","train_tfidf_processed.csv"))
        save_data(test_df,file_path=os.path.join("data/processed","test_tfidf_processed.csv"))
        
    except Exception as e:
        logger.error("the file is may be corrupted leads to : %s",e)
        raise
    
if __name__ == '__main__':
    main()
        
    
                