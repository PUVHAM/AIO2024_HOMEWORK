import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from config import DatasetConfig, ModelConfig
from data_utils import load_dataset
from models import RegressionMLP, ClassificationMLP, ImageClassificationMLP
from Module_5.Week_1.visualize import plot_figures

np.random.seed(DatasetConfig.RANDOM_SEED)
torch.manual_seed(DatasetConfig.RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(DatasetConfig.RANDOM_SEED)
    
def main():
    task = int(input("Choose task (1: Regression, 2: Classification, 3: Image_Classification): "))
    task_name = list(DatasetConfig.DATASET_PATH.keys())[task - 1]

    train_loader, val_loader, test_loader = load_dataset(task_name)
    hidden_dims = ModelConfig.model_type.get(task_name)["HIDDEN_DIMS"]

    if task == 1:
        # Task 1: Regression
        input_dims = next(iter(train_loader))[0].shape[1]
        output_dims = 1
        model = RegressionMLP(input_dims, hidden_dims, output_dims, task_name)
    elif task == 2:
        # Task 2: Classification
        input_dims = next(iter(train_loader))[0].shape[1]
        output_dims = len(set(d[1].item() for d in train_loader.dataset))
        model = ClassificationMLP(input_dims, hidden_dims, output_dims, task_name)
    elif task == 3:
        # Task 3: Image Classification
        input_dims = 128 * 128
        output_dims = len(os.listdir(DatasetConfig.IMAGE_CLASSIFICATION_DIR["Train"]))
        model = ImageClassificationMLP(input_dims, hidden_dims, output_dims, task_name)
    else:
        print("Invalid task.")
        return

    train_losses, val_losses, train_metrics, val_metrics = model.train_model(train_loader, val_loader)
        
    model.evaluate_model(val_loader, data_name="validation")
    model.evaluate_model(test_loader, data_name="test")
    
    if task in [2, 3]:
        plot_figures(train_losses, val_losses, train_metrics, val_metrics)
    else: 
        plot_figures(train_losses, val_losses, train_metrics, val_metrics, metric="R2")

if __name__ == "__main__":
    main()