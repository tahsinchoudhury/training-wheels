import torch
import torch.nn as nn
import os

from .model import LinearRegressionModel
from .synthetic_data_handler import load_data_from_csv
from . import sgd

class Trainer:
    def shuffle(self, X, y):
        torch.manual_seed(42)
        indices = torch.randperm(X.size(0))
        X = X[indices]
        y = y[indices]

        split_idx = int(0.8 * X.size(0))

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        return X_train, X_test, y_train, y_test
    
    def save(
            self, 
            model,
            model_dir="models"
    ):
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "linear_regression_model.pth")
        torch.save(model.state_dict(), model_path)

        print("Training complete")
        print("Weight:", model.weight.item())
        print("Bias:", model.bias.item())
        print("Model saved to:", model_path)

    def train(self):
        X, y = load_data_from_csv()
        X_train, X_test, y_train, y_test = self.shuffle(X, y)
        model = LinearRegressionModel()
        criterion = nn.MSELoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer = sgd.SGD(model.parameters(), lr=0.01)
        epochs = 1000
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            y_train_pred = model(X_train)
            train_loss = criterion(y_train_pred, y_train)

            train_loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = criterion(y_test_pred, y_test)

            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Train Loss: {train_loss.item():.6f} "
                f"Test Loss: {test_loss.item():.6f}"
            )
        
        self.save(model)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()