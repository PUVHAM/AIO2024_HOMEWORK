from config import DatasetConfig
from visualize import plot_figures
from load_dataset import load_df, split_dataset
from logistic_regression import LogisticRegression

def run_experiment(csv_name):
    x, y = load_df(csv_name, DatasetConfig.DATASET_PATH.get(csv_name))
    x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(x, y)
    regression = LogisticRegression(x_train, y_train, x_val, y_val, csv_name)
    train_losses, val_losses, train_accs, val_accs, theta = regression.train()
    plot_figures(train_losses, val_losses, train_accs, val_accs)
    
    val_set_acc = regression._compute_accuracy(x_val, y_val, theta)
    test_set_acc = regression._compute_accuracy(x_test, y_test, theta)
    print('Evaluation on validation and test set:')
    print(f'Validation Accuracy: {val_set_acc}')
    print(f'Test Accuracy: {test_set_acc}')
    
if __name__ == "__main__":
    run_experiment("Titanic")    