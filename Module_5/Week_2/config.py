import os
import numpy as np
from pathlib import Path

WEEK_DIR = Path(__file__).parent
MODULE_DIR = WEEK_DIR.parent

class DatasetConfig:
    # Data directories
    DATASET_DIR = os.path.join(MODULE_DIR, os.path.join(WEEK_DIR, 'Data'))
    
    # Data files
    DATASET_PATH = {
        "Credit": os.path.join(DATASET_DIR, 'creditcard.csv'),
        "Sentiment": os.path.join(DATASET_DIR, 'Twitter_Data.csv')
    }

    # Split dataset = 7:2:1
    RANDOM_SEED = 2
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    IS_SHUFFLE = True
    
class ModelConfig:
    # Hyperparameters
    model_type = {
        "Credit": {"LEARNING_RATE": 0.01, "EPOCHS": 30, "BATCH_SIZE": 1024},
        "Sentiment": {"LEARNING_RATE": 0.1, "EPOCHS": 200, "BATCH_SIZE": 65536},
    }