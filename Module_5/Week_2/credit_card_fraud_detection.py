import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
from config import DatasetConfig
from Module_5.Week_1.visualize import plot_figures
from load_dataset import load_df, split_dataset
from softmax_regression import SoftmaxRegression

def run_experiment(csv_name):
    x, y, n_classes = load_df(csv_name)
    x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(x, y, csv_name)
    regression = SoftmaxRegression(x_train, y_train, x_val, y_val, n_classes, csv_name)
    train_losses, val_losses, train_accs, val_accs, theta = regression.train()
    plot_figures(train_losses, val_losses, train_accs, val_accs)
    
    val_set_acc = regression._compute_accuracy(x_val, y_val, theta)
    test_set_acc = regression._compute_accuracy(x_test, y_test, theta)
    print('Evaluation on validation and test set:')
    print(f'Validation Accuracy: {val_set_acc}')
    print(f'Test Accuracy: {test_set_acc}')
    
if __name__ == "__main__":
    run_experiment("Credit")    