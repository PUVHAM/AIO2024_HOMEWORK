import pandas as pd
import numpy as np

from config import DatasetConfig
from preprocessing import one_hot_encoding, text_normalize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

import os
import gdown
from config import DatasetConfig

def download_dataset(csv_name):
    dataset_ids = {
        "Credit": "1_bnvfnPLzQFflb_SbchB6dXVqRPxMMfa",  
        "Sentiment": "1ZzHQ2DUshohMmUKtdLV5jIobz4QSZvYH"  
    }
    
    file_id = dataset_ids[csv_name]
    url = f'https://drive.google.com/uc?id={file_id}'
    
    os.makedirs(DatasetConfig.DATASET_DIR, exist_ok=True)
    
    output_path = DatasetConfig.DATASET_PATH[csv_name]
    
    gdown.download(url, output=output_path, quiet=True, fuzzy=True)
    print(f"Downloaded {csv_name} dataset to {output_path}")

def split_dataset(x, y, csv_name):
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=DatasetConfig.VAL_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED,
                                                      shuffle=DatasetConfig.IS_SHUFFLE)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                      test_size=DatasetConfig.TEST_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED,
                                                      shuffle=DatasetConfig.IS_SHUFFLE)
    if csv_name == "Credit":        
        scaler = StandardScaler()
        x_train[:, 1:] = scaler.fit_transform(x_train[:, 1:])
        x_val[:, 1:] = scaler.transform(x_val[:, 1:])
        x_test[:, 1:] = scaler.transform(x_test[:, 1:])
    return x_train, y_train, x_test, y_test, x_val, y_val

def load_df(csv_name):
    csv_path = DatasetConfig.DATASET_PATH[csv_name]
    if not os.path.exists(csv_path):
        try:
            download_dataset(csv_name)
        except Exception as e:
            ERROR_MSG = f"Failed to download the {csv_name} dataset. Please check the download process."
            raise e(ERROR_MSG)
        
    df = pd.read_csv(csv_path)
    if csv_name == "Credit":        
        dataset_arr = df.to_numpy().astype(np.float64)
        x, y = dataset_arr[:, :-1], dataset_arr[:, -1]
        n_classes, n_samples = np.unique(y, axis=0).shape[0], y.shape[0]
    elif csv_name == "Sentiment":
        df = df.dropna()
        df['normalized_text'] = df['clean_text'].apply(text_normalize)
        vectorizer = TfidfVectorizer(max_features=2000)
        x = vectorizer.fit_transform(df['normalized_text']).toarray()
        
        y = df['category'].to_numpy() + 1
        n_classes, n_samples = df['category'].nunique(), df['category'].size
    else:
        raise ValueError("Unsupported csv_name. Only 'Credit' and 'Sentiment' are allowed.")

    intercept = np.ones((x.shape[0], 1))
    x = np.concatenate((intercept, x), axis=1) 
    
    # One-hot encoding for y
    y = y.astype(np.uint8) 
    y = one_hot_encoding(y, n_classes, n_samples)
    return x, y, n_classes