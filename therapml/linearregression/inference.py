import os
import torch
import matplotlib.pyplot as plt

from .model import LinearRegressionModel
from .synthetic_data_handler import load_data_from_csv


class PredictionPlotter:
    def __init__(self, plot_dir: str = "plots"):
        self.plot_dir = plot_dir
        os.makedirs(self.plot_dir, exist_ok=True)

    def plot(
        self,
        X: torch.Tensor,
        y_preds: torch.Tensor,
        y_true: torch.Tensor = None,
        filename: str = "inference_plot.png"
    ) -> None:
        X_np = X.squeeze().numpy()
        y_preds_np = y_preds.squeeze().numpy()

        plt.figure()
        plt.scatter(X_np, y_preds_np, label="Predicted")

        if y_true is not None:
            y_true_np = y_true.squeeze().numpy()
            plt.scatter(X_np, y_true_np, label="True")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Model Predictions")
        plt.legend()

        plot_path = os.path.join(self.plot_dir, filename)
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved to {plot_path}")


class ModelInference:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = LinearRegressionModel()
        self._load_model()

    def _load_model(self) -> None:
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(X)


class InferenceApp:
    def __init__(
        self,
        model_path: str = os.path.join("models", "linear_regression_model.pth"),
        plot_dir: str = "plots"
    ):
        self.model_inference = ModelInference(model_path)
        self.plotter = PredictionPlotter(plot_dir)

    def run(self) -> None:
        X, y = load_data_from_csv()
        y_preds = self.model_inference.predict(X)
        self.plotter.plot(X, y_preds, y)


if __name__ == "__main__":
    app = InferenceApp()
    app.run()