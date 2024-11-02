import pandas as pd
import numpy as np

from config import DatasetConfig
from preprocessing import run_preprocess
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_dataset(x, y):
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=DatasetConfig.VAL_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED,
                                                      shuffle=DatasetConfig.IS_SHUFFLE)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train,
                                                      test_size=DatasetConfig.TEST_SIZE,
                                                      random_state=DatasetConfig.RANDOM_SEED,
                                                      shuffle=DatasetConfig.IS_SHUFFLE)
    scaler = StandardScaler()
    x_train[:, 1:] = scaler.fit_transform(x_train[:, 1:])
    x_val[:, 1:] = scaler.transform(x_val[:, 1:])
    x_test[:, 1:] = scaler.transform(x_test[:, 1:])
    return x_train, y_train, x_test, y_test, x_val, y_val

def load_df(csv_name, csv_path):
    index_col = {
        "Titanic": 'PassengerId',
        "Sentiment": 'id'
    }
    df = pd.read_csv(csv_path, index_col=index_col.get(csv_name))
    if csv_name == "Titanic":
        dataset_arr = df.to_numpy().astype(np.float64)
        x, y = dataset_arr[:, :-1], dataset_arr[:, -1]
        intercept = np.ones((x.shape[0], 1))
        x = np.concatenate((intercept, x), axis=1) 
    else:
        x, y = run_preprocess(df)
    return x, y