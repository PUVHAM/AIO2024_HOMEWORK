import matplotlib.pyplot as plt

def plot_figures(train_losses, val_losses, train_accs, val_accs, score_type="Accuracy"):
    _, ax = plt.subplots(2, 2, figsize=(12, 10))

    ax[0, 0].plot(train_losses)
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_title('Training Loss')

    ax[0, 1].plot(val_losses, color='orange')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Loss')
    ax[0, 1].set_title('Validation Loss')

    ax[1, 0].plot(train_accs)
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel(f'{score_type}')
    ax[1, 0].set_title(f'Training {score_type}')

    ax[1, 1].plot(val_accs, color='orange')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel(f'{score_type}')
    ax[1, 1].set_title(f'Validation {score_type}')

    plt.tight_layout()  
    plt.show()
