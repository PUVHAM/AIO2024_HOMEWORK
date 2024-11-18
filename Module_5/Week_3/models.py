import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from config import ModelConfig

class BaseMLP(nn.Module, ABC):
    def __init__(self, input_dims, hidden_dims, output_dims, model_name):
        super(BaseMLP, self).__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        
        self.config = ModelConfig.model_type.get(model_name)
        self.criterion = self._get_criterion()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.config["LEARNING_RATE"], 
            weight_decay=0, 
            momentum=0
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"{model_name} initialized on {self.device}")
        self.to(self.device)

    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def _get_criterion(self):
       pass
    
    @abstractmethod
    def train_model(self, train_loader, val_loader):
        pass

    @abstractmethod
    def evaluate_model(self, test_loader):
        pass

class RegressionMLP(BaseMLP):
    def __init__(self, input_dims, hidden_dims, output_dims, model_name):
        super().__init__(input_dims, hidden_dims, output_dims, model_name)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.output(x).squeeze(1)

    def _get_criterion(self):
        return nn.MSELoss()

    def train_model(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        train_r2 = []
        val_r2 = []
        
        self.epochs = self.config["EPOCHS"]

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_target = []
            train_predict = []

            self.train()
            for x_samples, y_samples in train_loader:
                x_samples = x_samples.to(self.device)
                y_samples = y_samples.to(self.device)

                self.optimizer.zero_grad()
                outputs = self(x_samples)
                train_predict += outputs.tolist()
                train_target += y_samples.tolist()

                loss = self.criterion(outputs, y_samples)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            train_r2.append(self.r_squared(train_target, train_predict))

            self.eval()
            val_loss = 0.0
            val_target = []
            val_predict = []

            with torch.no_grad():
                for x_samples, y_samples in val_loader:
                    x_samples = x_samples.to(self.device)
                    y_samples = y_samples.to(self.device)

                    outputs = self(x_samples)
                    val_predict += outputs.tolist()
                    val_target += y_samples.tolist()

                    loss = self.criterion(outputs, y_samples)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            val_r2.append(self.r_squared(val_target, val_predict))

            print(f"EPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}")
        
        return train_losses, val_losses, train_r2, val_r2
            
    def evaluate_model(self, data_loader, data_name="dataset"):
        self.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for x_samples, y_samples in data_loader:
                x_samples = x_samples.to(self.device)
                y_samples = y_samples.to(self.device)

                outputs = self(x_samples)
                y_pred.extend(outputs.cpu().tolist())
                y_true.extend(y_samples.cpu().tolist())

        r2_score = self.r_squared(y_true, y_pred)

        print(f"Evaluation on {data_name} set:")
        print(f"R2 score: {r2_score:}")
        
    @staticmethod
    def r_squared(y_true, y_pred):
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)
        mean_true = torch.mean(y_true)
        ss_tot = torch.sum((y_true - mean_true) ** 2)
        ss_res = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        return r2

class ClassificationMLP(BaseMLP):
    def __init__(self, input_dims, hidden_dims, output_dims, model_name):
        super(ClassificationMLP, self).__init__(input_dims, hidden_dims, output_dims, model_name)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.output(x).squeeze(1)
    
    def _get_criterion(self):
        return nn.CrossEntropyLoss()

    def train_model(self, train_loader, val_loader):
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        self.epochs = self.config["EPOCHS"]

        for epoch in range(self.epochs):
            train_loss = 0.0
            train_target = []
            train_predict = []

            self.train()
            for x_samples, y_samples in train_loader:
                x_samples = x_samples.to(next(self.parameters()).device)
                y_samples = y_samples.to(next(self.parameters()).device)


                self.optimizer.zero_grad()
                outputs = self(x_samples)
                loss = self.criterion(outputs, y_samples)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

                train_predict.append(outputs.detach().cpu())
                train_target.append(y_samples.detach().cpu())
            
            train_loss /= len(train_loader)
            train_losses.append(train_loss)
            
            train_predict = torch.cat(train_predict)
            train_target = torch.cat(train_target)
            train_accs.append(self.compute_accuracy(train_predict, train_target))

            val_loss = 0.0
            val_target = []
            val_predict = []

            self.eval()
            with torch.no_grad():
                for x_samples, y_samples in val_loader:
                    x_samples = x_samples.to(next(self.parameters()).device)
                    y_samples = y_samples.to(next(self.parameters()).device)

                    outputs = self(x_samples)
                    val_loss += self.criterion(outputs, y_samples).item()
                    val_predict.append(outputs.cpu())
                    val_target.append(y_samples.cpu())

            val_loss /= len(val_loader)
            val_losses.append(val_loss)
            
            val_predict = torch.cat(val_predict)
            val_target = torch.cat(val_target)
            val_accs.append(self.compute_accuracy(val_predict, val_target))

            print(f"EPOCH {epoch + 1}:\tTraining loss: {train_loss:.3f}\tValidation loss: {val_loss:.3f}")
            
        return train_losses, val_losses, train_accs, val_accs

    def evaluate_model(self, data_loader, data_name="dataset"):
        self.eval()
        all_targets = []
        all_predictions = []

        with torch.no_grad():
            for x_samples, y_samples in data_loader:
                x_samples = x_samples.to(self.device)
                y_samples = y_samples.to(self.device)
                outputs = self(x_samples)

                all_predictions.append(outputs.cpu())
                all_targets.append(y_samples.cpu())

        all_predictions = torch.cat(all_predictions)
        all_targets = torch.cat(all_targets)

        accuracy = self.compute_accuracy(all_predictions, all_targets)

        # Print result
        print(f"Evaluation on {data_name} set:")
        print(f"Accuracy: {accuracy}")


    @staticmethod
    def compute_accuracy(y_hat, y_true):
        _, y_hat = torch.max(y_hat, dim=1)
        correct = (y_hat == y_true).sum().item()
        accuracy = correct / len(y_true)
        return accuracy
    
class ImageClassificationMLP(ClassificationMLP):
    def __init__(self, input_dims, hidden_dims, output_dims, model_name):
        super(ImageClassificationMLP, self).__init__(input_dims, hidden_dims, output_dims, model_name)
        self.linear1 = nn.Linear(input_dims, hidden_dims*4)
        self.linear2 = nn.Linear(hidden_dims*4, hidden_dims*2)
        self.linear3 = nn.Linear(hidden_dims*2, hidden_dims)
        self.output = nn.Linear(hidden_dims, output_dims)
        self.relu = nn.ReLU()
        
        self.criterion = self._get_criterion()
        self.optimizer = torch.optim.SGD(
            self.parameters(), 
            lr=self.config["LEARNING_RATE"], 
            weight_decay=0, 
            momentum=0
        )
        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def forward(self, x):
        x = nn.Flatten()(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return self.output(x).squeeze(1)