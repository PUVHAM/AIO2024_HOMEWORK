import sys
import torch
import torchvision.transforms as transforms
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST

sys.path.append(str(Path(__file__).resolve().parents[2]))
from Module_5.Week_1.visualize import plot_figures
from Module_5.Week_3.models import ClassificationMLP
from Module_5.Week_3.config import ModelConfig
from Module_5.Week_3.main import get_seed


class FashionMNISTMLP(ClassificationMLP):
    def __init__(self, input_dims, hidden_dims, output_dims, model_name, optimizer):
        super(FashionMNISTMLP, self).__init__(input_dims, hidden_dims, output_dims, model_name)
        self.layer1 = nn.Linear(input_dims, hidden_dims)
        self.layer2 = nn.Linear(hidden_dims, hidden_dims)
        self.layer3 = nn.Linear(hidden_dims, hidden_dims)
        self.layer4 = nn.Linear(hidden_dims, hidden_dims)
        self.layer5 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.sigmoid = nn.Sigmoid()
        
        self.criterion = self._get_criterion()
        self.optimizer = optimizer(
            self.parameters(), 
            lr=self.config["LEARNING_RATE"], 
            betas=(0.9, 0.999),  
            eps=1e-8,       
            weight_decay=0
        )
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.layer1(x)
        x = self.sigmoid(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        x = self.layer4(x)
        x = self.sigmoid(x)
        x = self.layer5(x)
        x = self.sigmoid(x)
        out = self.output(x)
        return out
    
if __name__ == "__main__":
    get_seed(52)
    train_dataset = FashionMNIST(root='AIO2024_HOMEWORK/Module_5/Week_4/data', train=True, download=True, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    test_dataset = FashionMNIST(root='AIO2024_HOMEWORK/Module_5/Week_4/data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=512, num_workers=0)
    
    model = FashionMNISTMLP(input_dims=784, hidden_dims=128, output_dims=10, 
                            optimizer=torch.optim.Adam, model_name="FashionMNIST")
    
    train_losses, val_losses, train_metrics, val_metrics = model.train_model(train_loader, test_loader)
        
    model.evaluate_model(test_loader, data_name="test")
    
    plot_figures(train_losses, val_losses, train_metrics, val_metrics)

    