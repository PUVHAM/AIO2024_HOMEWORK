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
        "Titanic": os.path.join(DATASET_DIR, 'titanic_modified_dataset.csv'),
        "Sentiment": os.path.join(DATASET_DIR, 'sentiment_analysis.csv')
    }

    # Split dataset = 7:2:1
    RANDOM_SEED = 2
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    IS_SHUFFLE = True
    
class ModelConfig:
    # Hyperparameters

    model_type = {
        "Titanic": {"LEARNING_RATE": 0.01, "EPOCHS": 100, "BATCH_SIZE": 16},
        "Sentiment": {"LEARNING_RATE": 0.01, "EPOCHS": 200, "BATCH_SIZE": 128},
    }