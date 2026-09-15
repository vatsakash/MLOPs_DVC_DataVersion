import os 
import logging
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('puntk')


#ensure the logs directory exists 
log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

#setting up logger 
logger = logging.getLogger('preProcessing')
logger.setLevel('DEBUG')

console_logger = logging.StreamHandler()
console_logger.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_logger.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_logger)
logger.addHandler(file_handler)

def transform_text(text):
    """Transforms the input text by converting it to lowercase and then change into vectors by converting through tokens"""
    
    ps = PorterStemmer()
    #convert to lower case 
    text = text.lower()
    #Tokenize the text 
    text = nltk.word_tokenize(text)
    #remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    #remove stopwords and puctuation 
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #stem the words 
    text = [ps.stem(word) for word in text]
    #join the tokens back into a single string 
    return "".join(text)


def preprocess_df(df,text_column='text',target_column='target'):
    """preprocess the dataframe by encoding the target column,removing duplicates,and transforming the text column.
    """
    try:
        logger.debug('Starting preprocessing for the DataFrame')
        #Encode the target column 
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target column encoded')
        
        #Remove duplicate rows
        df = df.drop_duplicates(keep='first')
        logger.debug('Duplicates removed')
        
        #Apply text transformation to the specified text column 
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df 
    except KeyError as e:
        logger.error('Column not found: %s',e)
        raise 
    except Exception as e:
        logger.error('Error during text normalization: %s',e)
        raise
    
def main(text_column='text',target_column='target'):
    """Main function to load the raw data, preprocess it, and save the processed data."""
    try:
        #Fetch the data from data/raw.
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data is loaded sucessully')
        
        #transform the data
        train_processed_data = preprocess_df(train_data,text_column=text_column,target_column=target_column)
        test_processed_data = preprocess_df(test_data,target_column=target_column,text_column=text_column)
        
        #save the preprocessed data in /data/processed
        data_path = os.path.join("./data","interim")
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path,"train_processed.csv"),index=False)
        test_processed_data.to_csv(os.path.join(data_path,"test_processed.csv"),index=False)
        
        logger.debug('processed data sved to %s',data_path)
    except FileNotFoundError as e:
        logger.error('File not Found: %s',e)
    except pd.errors.EmptyDataError as e:
        logger.error('No Data: %s',e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s',e)
        raise
    
if __name__ == '__main__':
    main()
        
        