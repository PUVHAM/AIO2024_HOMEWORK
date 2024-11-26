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
        "Regression": os.path.join(DATASET_DIR, 'Auto_MPG_data.csv'),
        "Classification": os.path.join(DATASET_DIR, 'NonLinear_data.npy'),
        "Image_Classification": os.path.join(DATASET_DIR, 'FER-2013.zip')
    }
    
    IMAGE_CLASSIFICATION_DIR = {
        "Train": os.path.join(DATASET_DIR, 'FER-2013', 'train'),
        "Test": os.path.join(DATASET_DIR, 'FER-2013', 'test')   
    }

    # Split dataset = 7:2:1
    RANDOM_SEED = 59
    VAL_SIZE = 0.2
    TEST_SIZE = 0.125
    IS_SHUFFLE = True
    
class ModelConfig:
    # Hyperparameters
    model_type = {
        "Regression": {"LEARNING_RATE": 1e-2, "EPOCHS": 100, "HIDDEN_DIMS": 64},
        "Classification": {"LEARNING_RATE": 1e-1, "EPOCHS": 100, "HIDDEN_DIMS": 128},
        "Image_Classification": {"LEARNING_RATE": 1e-2, "EPOCHS": 40, "HIDDEN_DIMS": 64},
        "FashionMNIST": {"LEARNING_RATE": 0.01, "EPOCHS": 300, "HIDDEN_DIMS": 128}
    }